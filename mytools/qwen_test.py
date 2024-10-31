from openai import OpenAI
import os
import base64
import re
import json
from PIL import Image, ImageDraw, ImageFont

def draw_bounding_boxes(image_path, entities, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=30)  # 使用默认字体
    filtered_entities = []
    for entity in entities:
        bbox = entity.get("bboxes", [0, 0, 0, 0])
        if isinstance(bbox[0], list):
            pass
        else:
            bbox = [bbox]
        for box in bbox:
            # for i in range(len(box)):
            #     box[i] = float(box[i]) / 1000.
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

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

input_folder = "workdirs/sa_000000/images"
output_folder = "workdirs/processed_sam"
info_folder = "workdirs/info"

# 确保输出文件夹和信息文件夹存在
os.makedirs(output_folder, exist_ok=True)
os.makedirs(info_folder, exist_ok=True)

for image_name in os.listdir(input_folder):
    image_path = f"{input_folder}/{image_name}"
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    base64_image = encode_image(image_path)
    
    text_prompt = '''You are a helpful assistant. Given the image please analyze the following image and complete the following tasks:
    
    1. Generate a concise caption for the image so that another model can generate the image via the caption.
    2. Extract the main foreground entity categories presented both in the image and the caption.
    3. For each category, detect the bounding boxes for the main entities belongs to the category and decribe the entity breifly with several words. Caution:
        The detected entities should be of high confidence. 
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
    
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model="qwen-vl-max-0809", 
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": text_prompt}
                    ]}
                ]
            )
            # 提取json字符串
            content_str = completion.choices[0].message.content
            json_str = re.sub(r'^```json\s*|\s*```$', '', content_str, flags=re.DOTALL)

            # 解析 JSON 数据
            result_dict = json.loads(json_str)
            # 提取并打印生成的图片描述
            caption = result_dict.get("caption", "No caption available")
            print(f"Generated Caption: {caption}")

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
                print(f"Entity: {entity['entity']} Bounding Box: {entity['bboxes']}")
            
            # 绘制边界框并保存图像
            output_path = f"{output_folder}/{image_name}"
            draw_bounding_boxes(image_path, filtered_entities, output_path)
            print(f"Bounding boxes drawn on the image and saved to {output_path}")

            # 保存提取的信息到对应的文件，使用过滤后的 entities
            info_path = f"{info_folder}/{image_name.split('.')[0]}.json"
            result_dict["entities"] = filtered_entities  # 更新 entities 为过滤后的 entities
            with open(info_path, 'w') as info_file:
                json.dump(result_dict, info_file, indent=4)
            print(f"Extracted information saved to {info_path}")
            break  # 成功处理，跳出循环
        except Exception as e:
            print(f"Error processing image {image_path} on attempt {attempt + 1}: {e}")
            if attempt == max_attempts - 1:
                print(f"Failed to process image {image_path} after {max_attempts} attempts.")