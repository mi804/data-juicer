import json

def parse_jsonl(file_path, lines_limit=10000):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        print(f"Reading data from {file_path}")
        for line in file:
            if len(data_list) >= lines_limit:
                break
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {e}")
    return data_list


if __name__ == "__main__":
    file_path = '/mnt/nas1/zhanghong/project/data-juicer/data/laion/laion-high-resolution-datajuicer_input.jsonl'  # 替换为你的文件路径
    
    all_data = parse_jsonl(file_path)
    # 打印前1000条数据
    for i, data in enumerate(all_data[:1000]):
        print(json.dumps(data, indent=2))
    
    # 你可以在这里对 all_data 进行进一步处理
    # 例如：统计数据条数、筛选特定数据等
    print(f"Total number of entries: {len(all_data)}")