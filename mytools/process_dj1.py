import os
import glob


# 示例用法
directory_path = '/mnt/nas1/zhanghong/data/laion/laion-hr_dj_40/configs/'
config_files = []
for i in range(1, 119):  
    file_path = os.path.join(directory_path, f'data_juicer_filter_laion_hr_{i}.yaml')
    config_files.append(file_path)
    print(file_path)

processing = 95
split = processing - 1
for file in config_files[split:]:
    out_file = os.path.join('/mnt/nas1/zhanghong/data/laion/laion-hr_dj_40/', 'dj_hr_out_' + file.split('/')[-1].split('.')[0].split('_')[-1] + '.jsonl')
    if os.path.exists(out_file):
        print(f'{out_file} already exists')
        continue
    print(f'processing {file}')
    os.system(f"dj-process --config {file}")