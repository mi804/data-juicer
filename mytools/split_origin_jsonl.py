import json
import os

def split_jsonl_file(input_file, output_dir, output_prefix, lines_per_file=5000000):
    """
    将JSONL文件按指定行数分割成多个文件，并保存到指定目录。

    :param input_file: 输入的JSONL文件路径
    :param output_dir: 输出文件的目录
    :param output_prefix: 输出文件的前缀
    :param lines_per_file: 每个文件的行数
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_counter = 1
    line_counter = 0
    output_file = None

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line_counter % lines_per_file == 0:
                    if output_file:
                        output_file.close()
                    output_filename = os.path.join(output_dir, f"{output_prefix}_{file_counter}.jsonl")
                    output_file = open(output_filename, 'w', encoding='utf-8')
                    file_counter += 1
                    # if file_counter > 2:
                    #     break

                output_file.write(line)
                line_counter += 1

    finally:
        if output_file:
            output_file.close()

    print(f"文件分割完成，共生成 {file_counter - 1} 个文件。")

# 示例用法
input_file = '/mnt/nas1/zhanghong/project/data-juicer/data/laion/laion-high-resolution-datajuicer_input.jsonl'  # 输入的JSONL文件路径
output_dir = '/mnt/nas1/zhanghong/project/data-juicer/data/laion/datajuicer_input_hr'  # 输出文件的目录
# output_dir = '/mnt/nas1/zhanghong/project/data-juicer/data/laion/datajuicer_input_hr_test'  # 输出文件的目录

output_prefix = 'datajuicer_hr'    # 输出文件的前缀
lines_per_file = 500000
split_jsonl_file(input_file, output_dir, output_prefix, lines_per_file)