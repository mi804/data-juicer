# 可控数据生成——数据集制作
## 1. 下载数据集
[img2dataset](https://github.com/rom1504/img2dataset.git)

```shell
# laion-high-resolution
img2dataset --url_list data/laion-high-resolution --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format files\
           --output_folder data/laion/laion-high-resolution --processes_count 30 --thread_count 128 --image_size 1024\
            --resize_only_if_bigger=True --resize_mode="no" --skip_reencode=True \
             --save_additional_columns '["similarity","hash","punsafe","pwatermark","LANGUAGE"]' --enable_wandb True

# laion-aesthetics_v2_4.75
img2dataset --url_list data/aesthetics_v2_4.75 --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format files\
           --output_folder data/laion/aesthetics_v2_4.75 --processes_count 16 --thread_count 128 --image_size 1024\
            --resize_only_if_bigger=True --resize_mode="no" --skip_reencode=True \
             --save_additional_columns '["similarity","hash","punsafe","pwatermark","AESTHETIC_SCORE"]' --enable_wandb True
```
## 2. datajuicer处理
### 2.1 生成总jsonl文件
遍历下载的所有图像路径，写入`laion-high-resolution-datajuicer_input.jsonl`
``` shell
python generate_jsonl.py
```
### 2.2 划分总jsonl文件
将`laion-high-resolution-datajuicer_input.jsonl`划分为若干个子集，输出到`datajuicer_input_hr`下,命名格式为`datajuicer_hr_{i}.jsonl`，共**118**个子集，每个子集**50W**数据。 **每个自己的数据数量可以修改**。
``` shell
python split_origin_jsonl.py
```

### 2.3 生成datajuicer config
为以上的每个子集生成一个config，config里修改 `dataset_path`，`export_path` 和 `project_name`等字段。

``` shell
python generate_yamls.py
```
datajuicer配置：
```yaml
dataset_path: /mnt/nas1/zhanghong/project/data-juicer/data/laion/datajuicer_input_hr/datajuicer_hr_1.jsonl
eoc_special_token: <|__dj__eoc|>
export_path: /mnt/nas1/zhanghong/project/data-juicer/data/laion/laion-hr_dj_40/dj_hr_out_1.jsonl
image_key: image
image_special_token: <image>
np: 39
open_tracer: true
process:
- image_shape_filter:
    any_or_all: any
    min_height: 1024
    min_width: 1024
- image_aspect_ratio_filter:
    any_or_all: any
    max_ratio: 2.0
    min_ratio: 0.5
- image_nsfw_filter:
    any_or_all: any
    hf_nsfw_model: /mnt/nas1/zhanghong/project/data-juicer/checkpoints/nsfw
    mem_required: 1GB
    score_threshold: 0.0006
- image_watermark_filter:
    any_or_all: any
    hf_watermark_model: /mnt/nas1/zhanghong/project/data-juicer/checkpoints/watermark_detector
    mem_required: 500MB
    prob_threshold: 0.8
- image_aesthetics_filter:
    any_or_all: any
    hf_scorer_model: /mnt/nas1/zhanghong/project/data-juicer/checkpoints/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE
    max_score: 10
    mem_required: 1500MB
    min_score: 5
project_name: laion_hr_filter_1
text_keys: text

```

### 2.4 datajuicer 过滤
共118个任务串行执行，每个任务有多个进程。可以通过分割任务号，放在多个机器上分开跑。
``` shell
python process_dj.py
python process_dj1.py
```
### 2.5 datajuicer 输出合并
将经过datajuicer过滤的数据合并到一个jsonl，输出到`data/laion/datajuicer_output_hr`下，`merged_output.jsonl`为`aes>=5`的数据，`aes_6.jsonl`为`aes>=6`的数据。
```shell
python merge_output_from_datajuicer.py
```
### 2.6 Qwen2-VL数据增强
生成 `Caption` 和用于 grouning 的 `bbox` 与 `entity`，输出路径为：`data/laion/datajuicer_output_hr/aes6-caption-bboxbyqwen.jsonl`

通过`semaphore = manager.Semaphore(2)`限制API调用频率不超过2张图像每秒，设置环境变量`DASHSCOPE_API_KEY`为api key

```shell
python qwen_test_jsonl.py
```

数据示例：

```json
{"caption": "A sunset over a city with mountains in the foreground and a body of water in the background.", "entities": [{"entity": "mountain", "bboxes": [[0.0, 0.268, 0.835, 1.0]]}, {"entity": "city", "bboxes": [[0.394, 0.812, 0.722, 0.95]]}, {"entity": "body of water", "bboxes": [[0.314, 0.687, 0.84, 0.844]]}], "image": ["/mnt/nas1/zhanghong/data/laion/laion-high-resolution/00000/000000974.jpg"], "text": "", "__dj__stats__": {"aspect_ratios": [1.4981711778], "image_aesthetics_scores": [6.1788592339], "image_height": [1367], "image_nsfw_score": [0.000135663], "image_watermark_prob": [0.5577891469], "image_width": [2048]}, "image_id": "000000974"}
```

![](/mnt/nas1/zhanghong/project/data-juicer/data/laion/datajuicer_output_hr/draw/000000974.jpg)

### 2.7 sam2生成mask
对于单个样本，输入由Qwen2VL生成的bbox，输出对应的mask和mask score。mask score直接在jsonl中用list保存,`mask_scores`字段。而mask则保存在文件中，由`'mask_file'`字段指引。

```shell
python predict_mask_with_sam2.py
```
数据示例：
```json
{"caption": "A sunset over a city with mountains in the foreground and a body of water in the background.", "entities": [{"entity": "mountain", "bbox": [0.0, 0.268, 0.835, 1.0]}, {"entity": "city", "bbox": [0.394, 0.812, 0.722, 0.95]}, {"entity": "body of water", "bbox": [0.314, 0.687, 0.84, 0.844]}], "image": "/mnt/nas1/zhanghong/data/laion/laion-high-resolution/00000/000000974.jpg", "text": "", "__dj__stats__": {"aspect_ratios": [1.4981711778], "image_aesthetics_scores": [6.1788592339], "image_height": [1367], "image_nsfw_score": [0.000135663], "image_watermark_prob": [0.5577891469], "image_width": [2048]}, "image_id": "000000974", "mask_scores": [0.993648111820221, 0.9628720879554749, 0.96222984790802], "mask_file": "/mnt/nas1/zhanghong/data/laion/datajuicer_output_hr/sam2_mask/000000974.npy"}
```
![](/mnt/nas1/zhanghong/project/data-juicer/data/laion/datajuicer_output_hr/draw_sam2_mask/000000974.png)

# python files
* `generate_jsonl.py`：生成总jsonl文件
* `split_origin_jsonl.py`：划分总jsonl文件
* `generate_yamls.py`：生成datajuicer config
* `process_dj.py`：启动datajuicer进程
* `merge_output_from_datajuicer.py`：datajuicer 输出合并
* `generate_bbox_by_qwen.py`：生成bbox 和caption
* `parse_qwen2_jsonl.py`：提供了qwen输出文件的处理函数`parse_jsonl_file`
* `predict_mask_with_sam2.py`：sam2生成mask
* `test_jsonl.py`：测试jsonl文件
* `prepare_data_emu3.py`：将bbox和caption数据给emu3做生成。
* `parse_datalist_emu3.py`：将datalist.jsonl转为emu接受的格式：train.json
* `filter_qwen_results.py`：Qwen2-VL有一部分图像转RGB会出问题，过滤掉数据
* `qwen_stat.py`： 统计Qwen2-VL的生成数据的Entity分布。
* `data_juicer_filter_laion_hr_0.yaml`：自定义datajuicer config模板
