# -*- coding: utf-8 -*-

import json
import os
from tqdm import tqdm
import torch

output_path = '/mnt/nas1/zhanghong/data/laion/datajuicer_output_hr/emu3'

out_jsonl = f"{output_path}/list/datalist.jsonl"


os.makedirs(f"{output_path}/feature", exist_ok=True)
os.makedirs(f"{output_path}/list", exist_ok=True)

# 读取已处理的图像路径
processed_images = []
print('reading processed images...')
if os.path.exists(out_jsonl):
    with open(out_jsonl, 'r') as f:
        for line in f:
            processed_images.append(json.loads(line)['image_id'])
print(f'finished reading {len(processed_images)} processed images')
datalist = {
    "prefix": f"{output_path}/feature",
    "path_list": []
}

for name in processed_images[:50000]:
    feature = os.path.join(output_path, 'feature', f"{name}.pth")
    feature = torch.load(feature)
    print(feature['images'].shape)
    print()