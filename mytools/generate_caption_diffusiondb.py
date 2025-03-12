from openai import OpenAI, BadRequestError, RateLimitError
import os
import base64
import re
import json
from PIL import Image, ImageDraw, ImageFont
import multiprocessing
from multiprocessing import Manager
from tqdm import tqdm
import random
import time


def draw_bounding_boxes(image_path, entities, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=30)  # 使用默认字体
    for entity in entities:
        bbox = entity.get("bboxes", [0, 0, 0, 0])
        if isinstance(bbox[0], list):
            pass
        else:
            bbox = [bbox]
        for box in bbox:
            x1, y1, x2, y2 = box
            category = entity.get("entity", "Unknown category")
            width, height = image.size
            x1_pixel = int(x1 * width)
            y1_pixel = int(y1 * height)
            x2_pixel = int(x2 * width)
            y2_pixel = int(y2 * height)
            draw.rectangle([x1_pixel, y1_pixel, x2_pixel, y2_pixel], outline="red", width=3)
            draw.text((x1_pixel, y1_pixel), category, fill="red", font=font)
            
    image.save(output_path)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(sample, output_folder, output_jsonl, api_key, semaphore, draw=True, error_jsonl=None):
    with semaphore:
        image_path = sample['image'][0]
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        base64_image = encode_image(image_path)
        
        text_prompt = '''You are a professional image captioner. Please provide a comprehensive and detailed description of the following image, ensuring the inclusion of the following elements:
        
        - Main subjects and objects present in the image.
        - Key visual elements, including colors, shapes, textures that stand out.
        - Spatial relationships and composition, focusing on how elements are arranged and interact within the frame.
        - Notable background elements that contribute to the overall context or setting.
        
        Generate a caption according to the image so that another model can generate the image via the caption. Just return the string description, do not return anything else.
        '''
        

        if draw:
            draw = False if random.random() < 0.05 else True
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                completion = client.chat.completions.create(
                    model="qwen-vl-max-latest", 
                    messages=[
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": text_prompt}
                        ]}
                    ],
                    seed=random.randint(0, 1000)
                )
                # 提取json字符串
                content_str = completion.choices[0].message.content
                
                result_dict = {'text':content_str}
                sample.update(result_dict)
                with open(output_jsonl, 'a') as out_file:
                    out_file.write(json.dumps(sample) + '\n')
                break  # 成功处理，跳出循环
            except Exception as e:
                if isinstance(e, json.JSONDecodeError):
                    print(f"Decode Error when processing image {image_path} on attempt {attempt + 1}: {e}")
                    with open(error_jsonl, 'a') as out_file:
                        out_file.write(json.dumps(sample) + '\n')
                    break
                elif isinstance(e, RateLimitError):
                    # print(f"Rate exceed when processing image {image_path} on attempt {attempt + 1}: {e}")
                    time.sleep(2)
                    if attempt == max_attempts - 1:
                        print(f"Failed to process image {image_path} after {max_attempts} attempts.")
                    continue
                elif isinstance(e, BadRequestError):
                    print(f"Error processing image {image_path} on attempt {attempt + 1}: {e}")
                    with open(error_jsonl, 'a') as out_file:
                        out_file.write(json.dumps(sample) + '\n')
                    break
                else:
                    print(f"Error processing image {image_path} on attempt {attempt + 1}: {e}")
                    with open(error_jsonl, 'a') as out_file:
                        out_file.write(json.dumps(sample) + '\n')
                    break


def main():
    
    # output_folder = "/mnt/nas1/zhanghong/data/laion/datajuicer_output_hr"    
    # input_jsonl = os.path.join(output_folder, "aes_6.jsonl")
    # output_jsonl = os.path.join(output_folder, "aes6-caption-bboxbyqwen.jsonl")
    # error_jsonl = os.path.join(output_folder, "aes6_error_images.jsonl")

    output_folder = "/mnt/nas1/zhanghong/project/data-juicer/data/laion/diffusion_db/qwen_caption"    
    input_jsonl = os.path.join(output_folder, "generated_images_0312.jsonl")
    output_jsonl = os.path.join(output_folder, "captionbyqwen.jsonl")
    error_jsonl = os.path.join(output_folder, "error_images.jsonl")
    # 确保输出文件夹和信息文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    image_field = 'image'

    # 读取已处理的图像路径
    processed_images = set()
    print('reading processed images...')
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r') as f:
            for line in f:
                processed_images.add(json.loads(line)[image_field][0])
    print(f'finished reading {len(processed_images)} processed images')
    error_iamges = set()
    if os.path.exists(error_jsonl):
        with open(error_jsonl, 'r') as f:
            for line in f:
                error_iamges.add(json.loads(line)[image_field][0])
    print(f'find {len(error_iamges)} error images')
    # 读取输入jsonl文件
    samples = []
    reading_limit = 1000000
    with open(input_jsonl, 'r') as f:
        for line in f:
            sample = json.loads(line)
            image_id = sample[image_field][0]

            # 检查图像是否已经处理过
            if image_id in processed_images or image_id in error_iamges:
                # print(f"Image {image_path} has already been processed. Skipping...")
                continue
            samples.append(sample)
            if len(samples) % reading_limit == 0:
                break # 
    print(f"reading {len(samples)} new images")

    # 多进程处理
    num_processes = 50
    manager = Manager()
    semaphore = manager.Semaphore(50)  # 限制API调用频率不超过2张图像每秒
    api_key = os.getenv("DASHSCOPE_API_KEY")
    draw = True
    draw_path = os.path.join(output_folder, 'draw_new')
    if draw:
        os.makedirs(draw_path, exist_ok=True)
    # tqdm config
    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(samples)) as pbar:
            results = []
            for sample in samples:
                result = pool.apply_async(process_image, args=(sample, draw_path, output_jsonl, api_key, semaphore, draw, error_jsonl), callback=lambda _: pbar.update(1))
                results.append(result)
            for result in results:
                result.get()

if __name__ == "__main__":
    main()