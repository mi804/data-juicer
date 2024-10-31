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
from collections import Counter


if __name__ == "__main__":

    workdir = "/mnt/nas1/zhanghong/data/laion/datajuicer_output_hr"
    qwen2_jsonl = os.path.join(workdir, "aes6-caption-bboxbyqwen.jsonl")
    

    all_data = parse_jsonl_file(qwen2_jsonl)
    counter = Counter()
    for i, data in tqdm(enumerate(all_data), total=len(all_data), desc="Processing images"):  # 使用 tqdm 显示进度条
        image_id = data['image_id']
        image_file = data['image'][0]
        num_entities = len(data['entities'])
        if num_entities == 0:
            print(f"Image {image_file} has no entities. Skipping...")
            continue
        counter[num_entities] += 1
    # sort counter by key
    counter = dict(sorted(counter.items(), key=lambda x: x[0]))
    print(counter)