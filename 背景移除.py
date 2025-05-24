import os
import sqlite3
import cv2
import json
from matplotlib import pyplot as plt
from sympy import im
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm  # 添加 tqdm 进度条库

"""
Hyper parameters
"""
TEXT_PROMPT = "car. tire."
IMG_PATH = "notebooks/images/truck.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hq_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True


BASE_DIR = os.path.abspath("E:\\WorkSpace\\Python\\icloset")
IMAGES_PATH = os.path.join(BASE_DIR, 'fashion-dataset', 'images')
PROCESSED_IMAGES_PATH = os.path.join(BASE_DIR, 'fashion-dataset', 'processed_images_2')
SQLITE_PATH = os.path.join(BASE_DIR, 'fashion-dataset', 'styles.sqlite')
SIZE = 224

# Read data from sqlite
def read_sqlite():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM styles where score >= 90 and processed = 0")
    rows = cur.fetchall()
    data = [dict(row) for row in rows]
    conn.close()
    return data


# create output directory
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model, mask_threshold=-2, max_hole_area=800, max_sprinkle_area=300)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

def process_image(img_path, text_prompt):


    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # FIXME: figure how does this influence the G-DINO model
    if DEVICE == "cuda":
        torch.cuda.amp.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    # 取第一个蒙版
    masks = masks[0]

    # 生成的 mask 已经在变量 masks 中
    # 读取原始图像
    img = cv2.imread(img_path)

    # 保存蒙版图像
    mask_output_path = os.path.join(PROCESSED_IMAGES_PATH, f"{Path(img_path).stem}_mask.png")
    cv2.imwrite(mask_output_path, masks * 255)

    # 创建一个带透明背景的图像
    background_removed = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    background_removed[:, :, 3] = masks * 255  # 使用 mask 设置 alpha 通道

    # 保存移除背景的图像
    background_removed_output_path = os.path.join(PROCESSED_IMAGES_PATH, f"{Path(img_path).stem}_no_bg.png")
    cv2.imwrite(background_removed_output_path, background_removed)

def update_processed_status(item_id):
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE styles SET processed = 1 WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()

# 读取数据
data = read_sqlite()
# 处理每一张图片
for item in tqdm(data[:100], desc="Processing images"):
    if item["id"] != 1855:
        continue
    image_path = os.path.join(IMAGES_PATH, str(item['id']) + '.jpg')
    process_image(image_path, item["subCategory"].lower() + ".")
    # update_processed_status(item['id'])