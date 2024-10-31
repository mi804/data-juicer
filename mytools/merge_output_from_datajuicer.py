import json
import os

def read_jsonl(file_path):
    """读取 JSONL 文件并返回一个列表，每个元素是一个 JSON 对象"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def merge_jsonl_files(out_jsonls, out_stats):
    """合并两个 JSONL 文件列表中的对应行"""
    merged_data = []
    
    for jsonl_file, stats_file in zip(out_jsonls, out_stats):
        jsonl_data = read_jsonl(jsonl_file)
        stats_data = read_jsonl(stats_file)
        
        for jsonl_item, stats_item in zip(jsonl_data, stats_data):
            merged_item = {**jsonl_item, **stats_item}
            merged_item['image_id'] = merged_item['image'][0].split('/')[-1].split('.')[0]
            merged_data.append(merged_item)
        print('finished reading: ', jsonl_file)
    
    return merged_data

# 定义文件路径
total_outs = 118
out_jsonls = [f'/mnt/nas1/zhanghong/data/laion/laion-hr_dj_40/dj_hr_out_{i}.jsonl' for i in range(1, total_outs+1)]
out_stats = [f'/mnt/nas1/zhanghong/data/laion/laion-hr_dj_40/dj_hr_out_{i}_stats.jsonl' for i in range(1, total_outs+1)]
out_path = '/mnt/nas1/zhanghong/data/laion/datajuicer_output_hr'
all_out = 'merged_output.jsonl'
aes_6 = 'aes_6.jsonl'
# 合并数据
merged_data = merge_jsonl_files(out_jsonls, out_stats)

# 将合并后的数据写入新的 JSONL 文件
with open(os.path.join(out_path, all_out), 'w', encoding='utf-8') as outfile:
    for item in merged_data:
        outfile.write(json.dumps(item) + '\n')
print(f"finished {all_out}")

with open(os.path.join(out_path, aes_6), 'w', encoding='utf-8') as outfile:
    for item in merged_data:
        if item['__dj__stats__']['image_aesthetics_scores'][0] >= 6:
            outfile.write(json.dumps(item) + '\n')
print(f"finished {aes_6}")