import argparse
import json

import torch
from transformers import (
    CLIPTokenizer,
)

from tqdm.contrib import tzip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clip_id = "/root/autodl-tmp/ImageGeneration/AI-ModelScope/clip-vit-large-patch14"
TOKENIZER = CLIPTokenizer.from_pretrained(clip_id)


def calculate_num_tokens(sentence):
    return len(TOKENIZER(sentence)['input_ids'])


def main(args):
    metadata_list = []
    dinov2_list = []
    llava_list = []
    with open(args.nuscenes_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data['filename']=data['source']
            data.pop('source')
            metadata_list.append(data)
    with open(args.dino_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data['filename'] = data['source']
            data.pop('source')
            dinov2_list.append(data)
    with open(args.llava_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            llava_list.append(data)
    assert len(metadata_list) == len(dinov2_list) == len(
        llava_list), "Length of metadata, dinov2 and llava should be same"

    # 定义一个排序函数，按 a_list 的顺序进行排序
    order = {d['filename']: i for i, d in enumerate(metadata_list)}
    def sort_by_order(lst):
        return sorted(lst, key=lambda d: order[d['filename']])

    # 对 b_list 和 c_list 进行排序
    dinov2_list = sort_by_order(dinov2_list)
    llava_list = sort_by_order(llava_list)
    day_num=0
    night_num=0
    rainy_num=0
    for meta, dino, llava in tzip(metadata_list, dinov2_list, llava_list):
        assert meta['filename'] == dino['filename'] == llava['filename'], f"File name should be same,got " \
                                                                      f"{meta['filename']},{ dino['filename']},{llava['filename']}"
        text = ""
        # first, we need to identify the weather condition of this sample
        if "Night" in meta['prompt'] or 'night' in meta['prompt']:
            if 'rain' in meta['prompt'] or 'Rain' in meta['prompt']:
                key_label = 'rainy night'
            else:
                key_label = 'night'
            night_num+=1
        elif 'rain' in meta['prompt'] or 'Rain' in meta['prompt']:
            key_label = 'rainy day'
            rainy_num+=1
        else:
            key_label = 'sunny day'
            day_num+=1
        text += key_label + ","
        #text += meta['prompt']
        text += dino['dino_semantic_label'] + ","
        sentences = llava['description'].split('.')
        for sentence in sentences:
            if calculate_num_tokens(text + sentence) < 77:
                text += sentence + "."
        meta['text'] = text[:-1]
        meta['file_name'] = meta['filename']
        meta.pop('filename')
    print(f"we totally have {night_num} Night Frame")
    print(f"we totally have {rainy_num} Rainy Frame")
    print(f"we totally have {day_num} Day Frame")
    with open(args.nuscenes_path.replace('_nuscenes', ''), 'w') as file:
        for meta in metadata_list:
            file.write(json.dumps(meta) + '\n')
    print(f"{args.nuscenes_path.replace('_nuscenes', '')} saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL file.")
    parser.add_argument("--nuscenes-path","--n", type=str, required=True, help="Path to the metadata_nuscenes.jsonl")
    parser.add_argument("--dino-path","--d", type=str, required=True, help="Path to the metadata_dinov2.jsonl")
    parser.add_argument("--llava-path","--l", type=str, required=True, help="Path to the metadata_llava.jsonl")
    args = parser.parse_args()
    main(args)
