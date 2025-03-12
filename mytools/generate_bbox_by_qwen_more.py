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
        
        text_prompt = '''You are a helpful assistant. Given the image please analyze the following image and complete the following tasks:
        
        1. Generate a concise caption for the image so that another model can generate the image via the caption.
        2. Extract the main foreground entity categories presented both in the image and the caption.
        3. For each category, detect the bounding boxes and a point for the main entities belongs to the category and decribe the entity breifly with several words. Caution:
            The detected entities should be of high confidence and the entity should occupy the max area in the bbox.
            If the entity is hard to detect, or occupies only a small area of its bounding box, you must ignore it.
            If one entity is a part of another entity, you must ignore it. For example, you should ignore the wheels of a car, and the balcony of a house.
            Each entity should have only a bounding box in the format [x1, y1, x2, y2] represented using absolute pixel coordinates.
            Different entities may share the same name, but they should be treated as different entities.
        
        Please provide the results in JSON format as follows, which can be directly loads by json.loads() in Python:
        {
        "caption": "Generated caption",
        "entities": [
            {
            "entity": "brief description of entity 0",
            "bboxes": 
                [x1, y1, x2, y2],
            },
            {
            "entity": "brief description of entity 1",
            "bboxes": 
                [x1, y1, x2, y2],
            },
            {
            "entity": "brief description of entity 2",
            "bboxes": 
                [x1, y1, x2, y2],
            },
            ...
        ]
        }'''

        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                completion = client.chat.completions.create(
                    model="qwen-vl-max-0809", 
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
                json_str = re.sub(r'^```json\s*|\s*```$', '', content_str, flags=re.DOTALL)

                # 解析 JSON 数据
                result_dict = json.loads(json_str)
                # 提取并打印生成的图片描述
                caption = result_dict.get("caption", "")
                # print(f"Generated Caption: {caption}")

                # 提取并过滤实体信息
                entities = result_dict.get("entities", [])
                filtered_entities = [entity for entity in entities if "bboxes" in entity and isinstance(entity["bboxes"], list)]
                for entity in filtered_entities:
                    if isinstance(entity['bboxes'][0], list):
                        pass
                    else:
                        entity['bboxes'] = [entity['bboxes']]
                    for box in entity['bboxes']:
                        for i in range(len(box)):
                            box[i] = float(box[i]) / 1000.
                    # print(f"Entity: {entity['entity']} Bounding Box: {entity['bboxes']}")
                
                # 绘制边界框并保存图像
                if draw:
                    output_path = f"{output_folder}/{os.path.basename(image_path)}"
                    draw_bounding_boxes(image_path, filtered_entities, output_path)

                # 将处理结果写入输出jsonl文件

                result_dict.update(sample)
                with open(output_jsonl, 'a') as out_file:
                    out_file.write(json.dumps(result_dict) + '\n')
                
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
    
    output_folder = "/mnt/nas1/zhanghong/data/laion/datajuicer_output_hr"    
    aes = 5.9
    input_jsonl = os.path.join(output_folder, f"aes{aes}.jsonl")
    output_jsonl = os.path.join(output_folder, f"aes{aes}-caption-bboxbyqwen.jsonl")
    error_jsonl = os.path.join(output_folder, f"aes{aes}_error_images.jsonl")
    former_jsonls = [os.path.join(output_folder, "aes6-caption-bboxbyqwen.jsonl"), os.path.join(output_folder, "aes6_error_images.jsonl")]

    # 确保输出文件夹和信息文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 读取已处理的图像路径
    processed_images = set()
    print('reading processed files...')
    for former_jsonl in former_jsonls:
        if os.path.exists(former_jsonl):
            with open(former_jsonl, 'r') as f:
                for line in f:
                    processed_images.add(json.loads(line)['image_id'])
    print(f'finished reading {len(processed_images)} processed images')

    print('reading processed images...')
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r') as f:
            for line in f:
                processed_images.add(json.loads(line)['image_id'])
    print(f'finished reading {len(processed_images)} processed images')
    error_iamges = set()
    if os.path.exists(error_jsonl):
        with open(error_jsonl, 'r') as f:
            for line in f:
                error_iamges.add(json.loads(line)['image_id'])
    print(f'find {len(error_iamges)} error images')
    # 读取输入jsonl文件
    samples = []
    reading_limit = 1000000
    with open(input_jsonl, 'r') as f:
        for line in f:
            sample = json.loads(line)
            image_id = sample['image_id']

            # 检查图像是否已经处理过
            if image_id in processed_images or image_id in error_iamges:
                # print(f"Image {image_path} has already been processed. Skipping...")
                continue
            samples.append(sample)
            if len(samples) % reading_limit == 0:
                break # 
    print(f"reading {len(samples)} new images")

    # 多进程处理
    num_processes = 10
    manager = Manager()
    semaphore = manager.Semaphore(10)  # 限制API调用频率不超过2张图像每秒
    api_key = os.getenv("DASHSCOPE_API_KEY")
    draw = False
    draw_path = os.path.join(output_folder, 'draw')
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