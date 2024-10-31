# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from PIL import Image
import torch
from modelscope import snapshot_download
from parse_qwen2_jsonl import parse_jsonl_file
from tqdm import tqdm

local_repo_path = '/mnt/nas1/zhanghong/Codes/Emu3'
sys.path.append(os.path.join(local_repo_path))
from emu3.mllm.processing_emu3 import Emu3Processor
from emu3.tokenizer import Emu3VisionVQModel, Emu3VisionVQImageProcessor


def smart_resize(image, image_area: int = 720 * 720):
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))

    image = image.resize((tw, th))
    return image


def main():
    '''datatype:
    {
        "caption": "A ",
        "image": "pathxxx/image_id.jpg",
        "entities": [
            {
                "entity": "xxx",
                "bbox": [x1, y1, x2, y2]
            },
            ...
        ],
        "image_id": "xxxx",
        "__dj__stats__": {...},
        "text": "",
    }
    '''
    # model path
    model_path = snapshot_download('BAAI/Emu3-VisionTokenizer')
    data_path = '/mnt/nas1/zhanghong/data/laion/datajuicer_output_hr/aes6-caption-bboxbyqwen.jsonl'
    output_path = '/mnt/nas1/zhanghong/data/laion/datajuicer_output_hr/emu3'
    image_area = 720 * 720

    out_jsonl = f"{output_path}/list/datalist.jsonl"
    error_jsonl = f"{output_path}/list/error.jsonl"

    image_processor = Emu3VisionVQImageProcessor.from_pretrained(model_path)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(model_path, device_map="cuda:0")
    image_tokenizer.eval()

    os.makedirs(f"{output_path}/feature", exist_ok=True)
    os.makedirs(f"{output_path}/list", exist_ok=True)

    # 读取已处理的图像路径
    processed_images = set()
    print('reading processed images...')
    if os.path.exists(out_jsonl):
        with open(out_jsonl, 'r') as f:
            for line in f:
                processed_images.add(json.loads(line)['image_id'])
    print(f'finished reading {len(processed_images)} processed images')

    error_iamges = set()
    if os.path.exists(error_jsonl):
        with open(error_jsonl, 'r') as f:
            for line in f:
                error_iamges.add(json.loads(line)['image_id'])
    print(f'find {len(error_iamges)} error images')

    input_data = parse_jsonl_file(data_path)

    for inp in tqdm(input_data, total=len(input_data), desc="Processing images for Emu3"):
        name = inp["image_id"]
        if name in processed_images or name in error_iamges:
            # print(f"Image {name} has already been processed. Skipping...")
            continue
        try:
            prompt = inp["caption"]

            # preprocess bbox
            entities = inp["entities"]
            bbox_control_prompt = '\nPrecise location control:\n\n' + len(entities) * '{}.\n\n'

            bbox_controls = []
            for entity in entities:
                entities = inp["entities"]
                bbox = [int(x * 1000) for x in entity["bbox"]]
                bbox_controls.append(f'Entity: [{entity["entity"]}]; box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]')

            prompt += bbox_control_prompt.format(*bbox_controls)
            image = Image.open(inp["image"]).convert("RGB")
            image = smart_resize(image, image_area)

            image = image_processor(image, return_tensors="pt")["pixel_values"]
            with torch.no_grad():
                image = image.cuda()
                token_ids = image_tokenizer.encode(image)

            token_ids = token_ids.squeeze(0).cpu().numpy()
            data = {
                "name": name,
                "images": token_ids,
                "texts": prompt
            }

            torch.save(data, f"{output_path}/feature/{name}.pth")
        # datalist["path_list"].append(f"{name}.pth")

            with open(out_jsonl, 'a') as f:
                f.write(json.dumps({'image_id':name, 'texts': prompt}) + '\n')
        except Exception as e:
            print(f"Error processing image {name}: {e}")
            with open(error_jsonl, 'a') as f:
                f.write(json.dumps({'image_id':name}) + '\n')


if __name__ == "__main__":
    main()
