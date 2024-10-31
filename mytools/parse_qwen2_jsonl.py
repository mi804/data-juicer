import json

def parse_jsonl_file(jsonl_file_path, read_limit=None):
    '''
    return:
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
    with open(jsonl_file_path, 'r') as file:
        all_infos = []
        for line in file:
            # 解析每一行数据
            try:
                sample = json.loads(line)
                # 提取并打印实体信息
                all_entities = []
                entities = sample.get("entities", [])
                for entity in entities:
                    entity_description = entity.get("entity", "Unknown category")
                    bboxes = entity.get("bboxes", [])
                    for bbox in bboxes:
                        all_entities.append({'entity': entity_description, 'bbox': bbox})
                sample['entities'] = all_entities
                sample['image'] = sample['image'][0]
                all_infos.append(sample)
            except Exception as e:
                print(f"Error: {e}")
                continue
            if read_limit and len(all_infos) >= read_limit:
                break
        return all_infos

if __name__ == "__main__":
    jsonl_file_path = "/mnt/nas1/zhanghong/project/data-juicer/workdirs/laion-hr/laion-hr-bboxbyqwen.jsonl"
    all_infos = parse_jsonl_file(jsonl_file_path)
    