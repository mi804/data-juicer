import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from parse_qwen2_jsonl import parse_jsonl_file
import json
from tqdm import tqdm


if __name__ == "__main__":

    workdir = "/mnt/nas1/zhanghong/data/laion/datajuicer_output_hr"
    qwen2_jsonl = os.path.join(workdir, "aes6-caption-bboxbyqwen.jsonl")
    out_jsonl = os.path.join(workdir, "aes6-caption-bboxbyqwen-filterd.jsonl")
    

    # 读取已处理的图像路径
    processed_images = set()
    print('reading processed images...')
    if os.path.exists(out_jsonl):
        with open(out_jsonl, 'r') as f:
            for line in f:
                processed_images.add(json.loads(line)['image_id'])
    print(f'finished reading {len(processed_images)} processed images')


    all_data = []
    with open(qwen2_jsonl, 'r') as file:
        for line in file:
            all_data.append(json.loads(line))
    corrupted_count = 0
    for i, data in tqdm(enumerate(all_data), total=len(all_data), desc="Processing images"):  # 使用 tqdm 显示进度条
        image_id = data['image_id']
        image_file = data['image'][0]
        if image_id in processed_images:
            # print(f"Image {image_file} has already been processed. Skipping...")
            continue
        try:
            # Load image
            image = Image.open(image_file).convert("RGB")
            with open(out_jsonl, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            corrupted_count += 1
            continue
    print(f"Corrupted images: {corrupted_count}")
    print(f"Total images: {len(all_data)}")
    print(f"Processed images: {len(all_data) - corrupted_count}")
