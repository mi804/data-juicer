import os
import json
from multiprocessing import Pool, cpu_count

# 图像文件夹路径
image_folder = '/mnt/nas1/zhanghong/data/laion/laion-high-resolution'
# 输出 JSONL 文件路径
output_file = '/mnt/nas1/zhanghong/data/laion/laion-high-resolution-datajuicer_input.jsonl'

# 获取所有子文件夹的路径
def get_subdirectories(root_dir):
    subdirs = []
    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path):
            subdirs.append(full_path)
    return subdirs

# 获取单个目录下的图像文件路径
def get_image_paths_in_dir(dir_path):
    image_files = []
    print(f'starting {dir_path}')
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if entry.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.abspath(full_path))
    print(f'<<<<<<ending {dir_path}')
    return image_files

# 获取所有图像文件的绝对路径
def get_all_image_paths(image_folder, num_processes=cpu_count()):
    print(f"Starting to collect subdirectories from {image_folder} using {num_processes} processes...")
    subdirs = get_subdirectories(image_folder)
    print(f"Finished collecting subdirectories. Total subdirectories found: {len(subdirs)}")
    
    print(f"Starting to collect image paths from subdirectories using {num_processes} processes...")
    image_files = []
    with Pool(num_processes) as pool:
        for paths in pool.imap_unordered(get_image_paths_in_dir, subdirs):
            image_files.extend(paths)
    print(f"Finished collecting image paths. Total images found: {len(image_files)}")
    return image_files

# 生成 JSONL 数据
def generate_jsonl_data(image_path):
    return {
        'image': [image_path],
        'text': ''
    }

# 使用多进程生成 JSONL 数据
def generate_jsonl_with_multiprocessing(image_folder, output_file, num_processes=cpu_count()):
    print("Starting to generate JSONL data...")
    image_files = get_all_image_paths(image_folder, num_processes)
    
    print("Writing JSONL data to file...")
    with Pool(num_processes) as pool:
        jsonl_data = pool.map(generate_jsonl_data, image_files)
    
    with open(output_file, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')
    
    # 输出总样本数
    total_samples = len(jsonl_data)
    print(f"Total samples: {total_samples}")
    print(f"JSONL data has been saved to {output_file}")

if __name__ == '__main__':
    generate_jsonl_with_multiprocessing(image_folder, output_file)