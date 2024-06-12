import os
import shutil
import json
import argparse
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def distribute_images(args):
    folder_path = args.folder
    limits = {
        "sunny": args.sunny_max,
        "rainy": args.rainy_max,
        "night": args.night_max
    }

    with open(os.path.join(folder_path, 'metadata.jsonl'), 'r') as f:
        metadata_lines = f.readlines()

    categorized_metadata = defaultdict(list)
    for line in metadata_lines:
        data = json.loads(line)
        text = data['text'].lower()
        if text.startswith('sunny day'):
            categorized_metadata['sunny'].append(data)
        elif 'night' in text:
            categorized_metadata['night'].append(data)
        elif text.startswith('rainy'):
            categorized_metadata['rainy'].append(data)
    logging.info(f"Found {len(categorized_metadata['sunny'])} sunny images")
    logging.info(f"Found {len(categorized_metadata['night'])} night images")
    logging.info(f"Found {len(categorized_metadata['rainy'])} rainy images")
    total_folder = os.path.join(args.target_folder, 'total')
    create_folder_if_not_exists(total_folder)
    total_metadata_path = os.path.join(args.target_folder, 'metadata.jsonl')
    total_meta_data=[]
    for category, metadata_list in categorized_metadata.items():
        logging.info(f"Processing {category} images")
        output_folder = os.path.join(args.target_folder, category)
        create_folder_if_not_exists(output_folder)
        output_metadata_path = os.path.join(output_folder, 'metadata.jsonl')

        count = 0
        limit = limits[category]
        with open(output_metadata_path, 'w') as meta_file:
            for metadata in tqdm(metadata_list):
                if limit > 0 and count >= limit:
                    break
                img_src_path = os.path.join(folder_path, metadata['file_name'])
                img_dst_path = os.path.join(output_folder, metadata['file_name'])
                total_img_dst_path = os.path.join(total_folder, metadata['file_name'])
                if os.path.exists(img_src_path):
                    shutil.copy(img_src_path, img_dst_path)
                    shutil.copy(img_src_path, total_img_dst_path)
                    meta_file.write(json.dumps(metadata) + '\n')
                    total_meta_data.append(metadata)
                    count += 1
    logging.info(f"Total {len(total_meta_data)} images copied to {args.target_folder}")
    with open(total_metadata_path, 'w') as total_meta_file:
        for metadata in tqdm(total_meta_data):
            total_meta_file.write(json.dumps(metadata) + '\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribute images into folders based on their descriptions.")
    parser.add_argument("--folder", type=str, help="Path to the folder containing images and metadata.jsonl")
    parser.add_argument("--target-folder", type=str, help="Target Folder to save the distributed images")
    parser.add_argument("--sunny-max", type=int, default=10000,
                        help="Maximum number of images in the sunny folder (0 for no limit)")
    parser.add_argument("--rainy-max", type=int, default=10000,
                        help="Maximum number of images in the rainy folder (0 for no limit)")
    parser.add_argument("--night-max", type=int, default=10000,
                        help="Maximum number of images in the night folder (0 for no limit)")
    args = parser.parse_args()
    distribute_images(args)
