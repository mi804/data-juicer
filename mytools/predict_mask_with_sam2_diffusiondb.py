import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import List
from parse_qwen2_jsonl import parse_jsonl_file
import json
from tqdm import tqdm


np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def build_sam2_predictor(model_cfg, checkpoint, device='cuda'):
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def rel2abs(bbox: List, H, W):
    x0, y0, x1, y1 = bbox
    x0 = int(x0 * W)
    x1 = int(x1 * W)
    y0 = int(y0 * H)
    y1 = int(y1 * H)
    return [x0, y0, x1, y1]

def predict_image(predictor, image, bboxes, output_file):
    # set image
    input_bboxes = np.array(bboxes)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_bboxes,
        multimask_output=False,
    )
    if len(masks.shape) == 3:
        masks = np.expand_dims(masks, axis=0)
        scores = np.expand_dims(scores, axis=0)
    if output_file is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.squeeze(0), plt.gca(), random_color=True)
        for box in input_bboxes:
            show_box(box, plt.gca())
        plt.axis('off')
        plt.savefig(output_file)
        plt.close()
    return masks, scores
    # Predict masks

if __name__ == "__main__":
    sam2_checkpoint = "/mnt/nas1/zhanghong/project/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    workdir = "/mnt/nas1/zhanghong/data/laion/diffusion_db"
    qwen2_jsonl = os.path.join(workdir, "caption-bboxbyqwen.jsonl")
    out_jsonl = os.path.join(workdir, "caption-bboxbyqwen-maskbysam2.jsonl")
    error_jsonl = os.path.join(workdir, "sam2_err.jsonl")


    out_mask_path = os.path.join(workdir, "sam2_mask")
    draw_path = os.path.join(workdir, "draw_sam2_mask")
    os.makedirs(out_mask_path, exist_ok=True)
    os.makedirs(draw_path, exist_ok=True)
    
    draw = True

    # 读取已处理的图像路径
    processed_images = set()
    print('reading processed images...')
    if os.path.exists(out_jsonl):
        with open(out_jsonl, 'r') as f:
            for line in f:
                image_id = json.loads(line)['image'].split('/')[-1].split('.')[0]
                processed_images.add(image_id)
    print(f'finished reading {len(processed_images)} processed images')

    error_iamges = set()
    if os.path.exists(error_jsonl):
        with open(error_jsonl, 'r') as f:
            for line in f:
                image_id = json.loads(line)['image'].split('/')[-1].split('.')[0]
                error_iamges.add(image_id)
    print(f'find {len(error_iamges)} error images')



    all_data = parse_jsonl_file(qwen2_jsonl)
    predictor = build_sam2_predictor(model_cfg, sam2_checkpoint)
    for i, data in tqdm(enumerate(all_data), total=len(all_data), desc="Processing images"):  # 使用 tqdm 显示进度条
        image_file = data['image']
        image_id = image_file.split('/')[-1].split('.')[0]
        
        if image_id in processed_images or image_id in error_iamges:
            # print(f"Image {image_file} has already been processed. Skipping...")
            continue
        try:
            # Load image
            image = Image.open(image_file)
            image = np.array(image.convert("RGB"))

            # preprocess bbox
            entities = data['entities']
            bboxes = []
            for entity in entities:
                bbox = rel2abs(entity['bbox'], image.shape[0], image.shape[1])
                bboxes.append(bbox)
            
            # predict mask
            out_file = os.path.join(draw_path, f'{image_id}.png') if draw else None
            masks, scores = predict_image(predictor, image, bboxes, out_file)
            
            # postprocess masks and scores
            scores = scores.flatten().tolist()
            data['mask_scores'] = scores
            mask_file = os.path.join(out_mask_path, f"{image_id}.npy")
            np.save(mask_file, masks.astype(np.bool_))
            data['mask_file'] = mask_file
            assert len(entities) == len(scores)

            # save to jsonl
            with open(out_jsonl, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            with open(error_jsonl, 'a') as f:
                f.write(json.dumps(data) + '\n')
