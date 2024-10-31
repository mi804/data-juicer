import yaml
import os

# 读取原始配置文件
with open('/mnt/nas1/zhanghong/data/laion/laion-hr_dj_40/configs/data_juicer_filter_laion_hr_0.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 复制并修改配置文件
for i in range(1, 119):
    # 复制原始配置
    new_config = config.copy()
    
    # 修改 'dataset_path' 字段
    new_config['dataset_path'] = '/mnt/nas1/zhanghong/data/laion/datajuicer_input_hr/datajuicer_hr_{}.jsonl'.format(i)
    
    # 修改 'export_path' 字段
    new_config['export_path'] = '/mnt/nas1/zhanghong/data/laion/laion-hr_dj_40/dj_hr_out_{}.jsonl'.format(i)

    new_config['project_name'] = 'laion_hr_filter_{}'.format(i)

    
    # 保存新的配置文件
    new_file_name = '/mnt/nas1/zhanghong/data/laion/laion-hr_dj_40/configs/data_juicer_filter_laion_hr_{}.yaml'.format(i)
    with open(new_file_name, 'w') as file:
        yaml.dump(new_config, file, default_flow_style=False)
    
    print('Generated {}'.format(new_file_name))

print('All files generated successfully.')