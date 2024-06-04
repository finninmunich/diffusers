import glob
import os
from pathlib import Path
import torch
from ddpm_inversion.inversion_utils import get_variance
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline

from pnp_utils import *

# suppress partial model loading warning
logging.set_verbosity_error()


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class PNP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        sd_version = config["sd_version"]

        if sd_version == '2.1':
            self.model_key = "/home/turing/cfs_cz/finn/codes/DrivingEdition/examples/text_to_image/stable-diffusion-2-1"
        elif sd_version == '1.5':
            # model_key = "/home/turing/cfs_cz/finn/codes/DrivingEdition/examples/text_to_image/stable-diffusion-v1-5"
            self.model_key = "/root/autodl-tmp/ImageGeneration/AI-ModelScope/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(self.model_key, ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print('SD model loaded')

        self.text_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]

        # feature extraction
        self.etas = config['etas']
        self.cfg_scale_src = config['cfg_scale_src']
        self.cfg_scale_tar = config['cfg_scale_tar']
        self._steps = config["steps"]
        self._save_steps = config["save_steps"]
        self._extract_reverse = config["extract_reverse"]
        self.inversion_func = self.ddim_inversion
        self.inverse_scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler")
        self.config = config
        self.noisy_latents = {}
        self.zs = {}

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self):
        # load image
        image = Image.open(self.config["image_path"]).convert('RGB')
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        # get noise
        if not self.noisy_latents:
            raise ValueError("No noisy latents found")
        # latents_path = os.path.join(self.config["latents_path"],
        #                             os.path.splitext(os.path.basename(self.config["image_path"]))[0],
        #                             f'noisy_latents_{self.scheduler.timesteps[0]}.pt')
        noisy_latent = self.noisy_latents[int(self.scheduler.timesteps[0])]
        return image, noisy_latent

    @torch.no_grad()
    def denoise_step_ddpm_inversion(self, x, t):
        # TODO: here the denoise_step should be modifed by "inversion_reverse_process"
        # mainly focused on the injection of noisy_latents and zt
        # shape of x [1,4,64,113]
        # register the time step and features in pnp injection modules
        batch_size = x.shape[0]
        source_latents = self.noisy_latents[int(t)]
        z = self.zs[int(t)]
        z = z.expand(batch_size,-1,-1,-1)
        latent_model_input = torch.cat([source_latents] + ([x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        # 1. get previous step value (=t-1)
        prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (source_latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self.scheduler._get_variance(timestep, prev_timestep)
        variance = get_variance(self, t)  # , prev_timestep)
        # Take care of asymetric reverse process (asyrp)
        model_output_direction = noise_pred
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
        pred_sample_direction = (1 - alpha_prod_t_prev - 1 * variance) ** (0.5) * model_output_direction
        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        # 8. Add noice if eta > 0
        if z is None:
            z = torch.randn(noise_pred.shape, device=self.device)
        sigma_z = 1 * variance ** (0.5) * z
        prev_sample = prev_sample + sigma_z
        return prev_sample

    @torch.no_grad()
    def denoise_step(self, x, t):
        # TODO: here the denoise_step should be modifed by "inversion_reverse_process"
        # mainly focused on the injection of noisy_latents and zt

        # register the time step and features in pnp injection modules
        source_latents = self.noisy_latents[int(t)]
        latent_model_input = torch.cat([source_latents] + ([x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True,
                       timesteps_to_save=None):
        timesteps = reversed(self.inverse_scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.inverse_scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.inverse_scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.inverse_scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    self.noisy_latents[int(t)] = latent
                    # torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        # torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return latent

    @torch.no_grad()
    def ddim_sample(self, x, cond, save_path, save_latents=False, timesteps_to_save=None):
        timesteps = self.inverse_scheduler.timesteps
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(x.shape[0], 1, 1)
                alpha_prod_t = self.inverse_scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.inverse_scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.inverse_scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps

            if save_latents:
                self.noisy_latents[int(t)] = x
                # torch.save(x, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return x

    @torch.no_grad()
    def extract_latents_ddpm_inversion(self, num_steps, data_path, save_path, timesteps_to_save,
                                       inversion_prompt='', extract_reverse=False):
        self.noisy_latents.clear()
        self.inverse_scheduler.set_timesteps(num_steps)
        timesteps = self.inverse_scheduler.timesteps.to(self.device)
        uncond_cond = self.get_text_embeds(inversion_prompt, "").to(self.device)
        uncond_embedding = uncond_cond[0].unsqueeze(0)
        text_embeddings = uncond_cond[1].unsqueeze(0)
        image = self.load_img(data_path)
        x0 = self.encode_imgs(image)  # x_0 with shape (1,4,64,113)
        variance_noise_shape = (
            self._steps,  # equals to num_diffusion_steps = 100
            self.unet.in_channels,
            x0.shape[-2],
            x0.shape[-1])
        etas = [self.etas] * self.inverse_scheduler.num_inference_steps

        # sample xts from x0
        alpha_bar = self.inverse_scheduler.alphas_cumprod
        sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)} # t_to_idx: [991:0,981:1,...1:99]
        xts = torch.zeros(
            (self._steps + 1, self.unet.in_channels, x0.shape[-2], x0.shape[-1])).to(x0.device)
        xts[0] = x0
        for t in reversed(timesteps):  # [1,11,21,31,...981,991]
            idx = self._steps - t_to_idx[int(t)]  # t=1,idx=1,t=11,idx=2,...t=991,idx=100
            xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        # Here the XTS contains x_0 -> x_100 in order
        # xts = sample_xts_from_x0(self, x0, num_inference_steps=self._steps)
        alpha_bar = self.inverse_scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=self.device)

        # xt = x0
        # op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
        t_to_idx = {int(v): k for k, v in enumerate(timesteps)} # t_to_idx: [991:0,981:1,...1:99]
        op = tqdm(timesteps)
        for t in op:
            # idx = t_to_idx[int(t)]
            idx = self._steps - t_to_idx[int(t)] - 1  # t: 991->idx: 99, 981->98, 971->97... 1->0
            # 1. predict noise residual
            xt = xts[idx + 1][None]  # +1 because we have x0 in the 0 position
            # xt = xts_cycle[idx+1][None]
            out = self.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
            if not inversion_prompt == "":
                cond_out = self.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)

            if not inversion_prompt == "":
                ## classifier free guidance
                noise_pred = out.sample + self.cfg_scale_src * (cond_out.sample - out.sample)
            else:
                noise_pred = out.sample
            # xtm1 =  xts[idx+1][None]
            xtm1 = xts[idx][None]  # (4,64,113) -> (1,4,64,113)
            # pred of x0
            pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5

            # direction to xt
            prev_timestep = t - self.inverse_scheduler.config.num_train_timesteps // self.inverse_scheduler.num_inference_steps
            alpha_prod_t_prev = self.inverse_scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else self.inverse_scheduler.final_alpha_cumprod

            variance = get_variance(self, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (0.5) * noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
            #self.noisy_latents[int(t)] = mu_xt
            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)  # xt-1 - mu_xt / sigma_t
            zs[idx] = z
            self.zs[int(t)] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx] = xtm1
            self.noisy_latents[int(t)] = xtm1
        # for self.noisy_latents : x0 -> x100
        # for self.zs: z99 -> z0
        # but since it is a dict, so order doesn't matter
        if not zs is None:
            zs[0] = torch.zeros_like(zs[0])
            self.zs[1] = zs[0]
    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, save_path, timesteps_to_save,
                        inversion_prompt='', extract_reverse=False, reconstruction=False):
        self.noisy_latents.clear()
        self.inverse_scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path)
        latent = self.encode_imgs(image)
        inverted_x = self.inversion_func(cond, latent, save_path, save_latents=not extract_reverse,
                                         timesteps_to_save=timesteps_to_save)
        # print(self.noisy_latents.keys())
        if reconstruction:
            # TODO: here the ddim_sample should be repalced by "our_inv"
            latent_reconstruction = self.ddim_sample(inverted_x, cond, save_path, save_latents=extract_reverse,
                                                     timesteps_to_save=timesteps_to_save)
            rgb_reconstruction = self.decode_latents(latent_reconstruction)

            return rgb_reconstruction  # , latent_reconstruction
        else:
            return None

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def feature_extraction_ddpm_inversion(self, ):
        # timesteps to save
        toy_scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler")
        toy_scheduler.set_timesteps(self.config['save_steps'])
        timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler,
                                                               num_inference_steps=self.config['save_steps'],
                                                               strength=1.0,
                                                               device=self.device)
        # timesteps: (991,981 -> 31,21,11,1)
        self.extract_latents_ddpm_inversion(data_path=self.config["image_path"],
                                            num_steps=self._steps,  # self.config['steps']
                                            save_path=self.config['output_path'],
                                            timesteps_to_save=timesteps_to_save,
                                            inversion_prompt=" ",
                                            extract_reverse=self._extract_reverse,
                                            )

    def feature_extraction(self, reconstruction=False):
        # timesteps to save
        toy_scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler")
        toy_scheduler.set_timesteps(self.config['save_steps'])
        timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler,
                                                               num_inference_steps=self.config['save_steps'],
                                                               strength=1.0,
                                                               device=self.device)
        # timesteps: (991,981 -> 31,21,11,1)
        recon_image = self.extract_latents(data_path=self.config["image_path"],
                                           num_steps=self._steps,  # self.config['steps']
                                           save_path=self.config['output_path'],
                                           timesteps_to_save=timesteps_to_save,
                                           inversion_prompt=" ",
                                           extract_reverse=self._extract_reverse,
                                           reconstruction=reconstruction)
        if reconstruction:
            T.ToPILImage()(recon_image[0]).save(os.path.join(self.config['output_path'], f'recon.jpg'))

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp_ddpm_inversion(self):
        _, self.eps = self.get_data()
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        edited_img = self.sample_loop_ddpm_inversion(self.eps)

    def sample_loop_ddpm_inversion(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step_ddpm_inversion(x, t)

            decoded_latent = self.decode_latent(x)
            T.ToPILImage()(decoded_latent[0]).save(
                f'{self.config["output_path"]}/output-{self.config["prompt"][:20]}.png')

        return decoded_latent

    def run_pnp(self):
        # load image
        # TODO: here self.eps should be replaced by x_t
        _, self.eps = self.get_data()
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        edited_img = self.sample_loop(self.eps)

    def sample_loop(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)

            decoded_latent = self.decode_latent(x)
            T.ToPILImage()(decoded_latent[0]).save(
                f'{self.config["output_path"]}/output-{self.config["prompt"][:20]}.png')

        return decoded_latent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_pnp.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config["output_path"], exist_ok=True)
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)

    seed_everything(config["seed"])
    print(config)
    pnp = PNP(config)
    pnp.feature_extraction_ddpm_inversion()
    #pnp.run_pnp_ddpm_inversion()
    # pnp.feature_extraction(reconstruction=False)
    pnp.run_pnp()
