"""
DeepFashion2 蒙版精细化脚本 V3

完全基于 Grounded-SAM-2 背景移除.py 的实现
迁移所有功能到 DeepFashion2 数据集
"""

import os
import json
import argparse
import traceback
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm


# ==================== 配置参数 ====================
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hq_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据集路径
if os.environ["colab"] == 1:
    DATASET_ROOT = "/content"
else:
    DATASET_ROOT = r"E:\WorkSpace\Python\icloset_train\deepfashion2_original_images"
TRAIN_IMAGE_DIR = os.path.join(DATASET_ROOT, "train", "image")
TRAIN_ANNOS_DIR = os.path.join(DATASET_ROOT, "train", "annos")
OUTPUT_DIR = os.path.join(DATASET_ROOT, "refine_annos_v3")
OUTPUT_MASKS_DIR = os.path.join(DATASET_ROOT, "refine_masks_v3")
OUTPUT_CROPPED_DIR = os.path.join(DATASET_ROOT, "refine_cropped_v3")
SQLITE_DB_PATH = os.path.join(DATASET_ROOT, "train.sqlite")

# 裁剪图像尺寸
SIZE = 224

# 过滤条件
FILTER_SOURCE = "shop"
FILTER_SCALES = [2, 3]
FILTER_OCCLUSION = 1
FILTER_VIEWPOINTS = [1, 2]

# 类别提示词表
CATEGORY_PROMPTS = {
    "short sleeve top": ["short sleeve top", "t-shirt", "tee", "短袖上衣", "短袖T恤"],
    "long sleeve top": ["long sleeve top", "sweater", "long-sleeve shirt", "长袖上衣", "毛衣", "长袖衬衫"],
    "short sleeve outwear": ["short sleeve outwear", "shrug", "短袖外套"],
    "long sleeve outwear": ["coat", "jacket", "blazer", "trench coat", "长袖外套", "风衣", "夹克"],
    "vest": ["vest", "sleeveless top", "马甲", "背心"],
    "sling": ["cami", "camisole", "吊带"],
    "shorts": ["shorts", "热裤", "短裤"],
    "trousers": ["trousers", "pants", "牛仔裤", "西裤"],
    "skirt": ["skirt", "裙子", "半身裙", "pleated skirt"],
    "short sleeve dress": ["short sleeve dress", "短袖连衣裙"],
    "long sleeve dress": ["long sleeve dress", "长袖连衣裙"],
    "vest dress": ["sleeveless dress", "背心裙"],
    "sling dress": ["cami dress", "slip dress", "吊带裙"]
}


# ==================== 初始化模型 ====================
# CUDA优化设置
if DEVICE == "cuda":
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# 构建SAM2
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model, mask_threshold=-2)

# 构建Grounding DINO
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


# ==================== 工具函数 ====================
def init_database(db_path: str):
    """初始化SQLite数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS refined_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            cropped_image_path TEXT NOT NULL,
            original_category TEXT NOT NULL,
            category TEXT,
            subcategory TEXT,
            season TEXT,
            scene TEXT,
            processed INTEGER DEFAULT 0
        )
    """)
    
    # 创建索引以提高查询效率
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_image_id ON refined_items(image_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_processed ON refined_items(processed)
    """)
    
    conn.commit()
    conn.close()
    print(f"数据库初始化完成: {db_path}")


def insert_refined_item(
    db_path: str,
    image_id: str,
    item_id: str,
    cropped_image_path: str,
    original_category: str
):
    """插入精细化后的item记录到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO refined_items 
        (image_id, item_id, cropped_image_path, original_category, processed)
        VALUES (?, ?, ?, ?, 0)
    """, (image_id, item_id, cropped_image_path, original_category))
    
    conn.commit()
    conn.close()


def check_record_exists(
    db_path: str,
    image_id: str,
    item_id: str
) -> bool:
    """检查数据库中是否已存在该记录"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM refined_items 
        WHERE image_id = ? AND item_id = ?
    """, (image_id, item_id))
    
    count = cursor.fetchone()[0]
    conn.close()
    
    return count > 0


def load_annotation(json_path: str) -> Dict:
    """加载JSON标注"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_annotation(json_path: str, data: Dict):
    """保存JSON标注"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def should_process_item(item: Dict, source: str) -> bool:
    """判断item是否符合处理条件"""
    return (source == FILTER_SOURCE and
            item.get("scale") in FILTER_SCALES and
            item.get("occlusion") == FILTER_OCCLUSION and
            item.get("viewpoint") in FILTER_VIEWPOINTS)


def fill_holes(mask, min_hole_area=5000):
    """
    填充蒙版中的小空洞，只填充面积小于阈值的空洞
    
    参数:
    mask: 输入的蒙版
    min_hole_area: 小于此面积的空洞将被填充
    """
    # 转换为二值图像
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    # 找到所有轮廓，包括内部空洞
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建一个复制的蒙版用于填充
    filled_mask = binary_mask.copy()
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        # 遍历所有轮廓
        for i, (contour, h) in enumerate(zip(contours, hierarchy)):
            # h[3] >= 0 意味着这是一个内部轮廓（洞）
            if h[3] >= 0:  # 这是一个洞
                area = cv2.contourArea(contour)
                if area < min_hole_area:  # 只填充小洞
                    cv2.drawContours(filled_mask, [contour], 0, (255,), -1)
    
    # 转换回原始数据类型和值范围
    return (filled_mask > 0).astype(mask.dtype)


def remove_isolated_spots(mask, max_spot_area=500):
    """
    去除蒙版中的孤立脏点，仅保留面积小于阈值的孤立区域
    
    参数:
    mask: 输入的蒙版
    max_spot_area: 大于此面积的孤立区域将被去除
    """
    # 转换为二值图像
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    # 标记连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # 创建一个新蒙版，初始为全0
    cleaned_mask = np.zeros_like(binary_mask)
    
    # 找到最大的连通区域（通常是主要物体）
    largest_label = 0
    largest_area = 0
    for i in range(1, num_labels):  # 从1开始，跳过背景(0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i
    
    # 将最大连通区域保留在新蒙版中
    cleaned_mask[labels == largest_label] = 255
    
    # 对于其他连通区域，仅保留面积小于阈值的区域
    for i in range(1, num_labels):
        if i != largest_label:
            area = stats[i, cv2.CC_STAT_AREA]
            if area < max_spot_area:  # 小于阈值的脏点保留
                cleaned_mask[labels == i] = 255
    
    # 转换回原始数据类型和值范围
    return (cleaned_mask > 0).astype(mask.dtype)


def mask_to_polygon(mask: np.ndarray, simplify_tolerance: float = 2.0) -> List[List[int]]:
    """将蒙版转换为多边形"""
    binary_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        epsilon = simplify_tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        polygon = approx.reshape(-1, 2).flatten().tolist()
        polygons.append(polygon)
    
    return polygons


# ==================== 核心处理函数 ====================
def process_image(img_path: str, text_prompt: str) -> np.ndarray:
    """
    处理单张图片，使用Grounding DINO + SAM2
    完全复刻背景移除.py的实现
    
    返回: 精细化的mask
    """
    # 使用grounding_dino的load_image函数
    image_source, image = load_image(img_path)

    # 设置SAM2图像
    sam2_predictor.set_image(image_source)

    # Grounding DINO检测
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    # 处理box坐标用于SAM2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # SAM2预测
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # 后处理mask
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # 取第一个蒙版
    if len(masks) > 0:
        mask = masks[0]
    else:
        # 如果没有检测到，返回空mask
        return np.zeros((h, w), dtype=np.uint8)

    # 填充mask中的空洞
    mask = fill_holes(mask)

    # 去除孤立脏点
    mask = remove_isolated_spots(mask)

    return mask


def process_single_item(
    image_source: np.ndarray,
    item: Dict,
    image_id: str,
    item_id: str,
    db_path: str
) -> Tuple[bool, List[List[int]]]:
    """
    处理单个item，包括保存mask和裁剪图像，以及写入数据库
    
    返回: (是否成功, 精细化后的segmentation)
    """
    # 优先检查数据库记录是否已存在
    if check_record_exists(db_path, image_id, item_id):
        # 记录已存在，直接返回成功
        segmentation = item.get("segmentation", [])
        return True, segmentation if segmentation else []
    
    category_name = item.get("category_name", "")
    
    # 生成输出文件名
    cropped_filename = f"{image_id}_{item_id}_cropped.png"
    cropped_output_path = os.path.join(OUTPUT_CROPPED_DIR, cropped_filename)
    
    # 检查裁剪图片是否已存在（但数据库中没有记录的情况）
    if os.path.exists(cropped_output_path):
        # 图片已存在，跳过处理但需插入数据库记录
        try:
            segmentation = item.get("segmentation", [])
            # 插入数据库记录（只保存文件名）
            insert_refined_item(
                db_path=db_path,
                image_id=image_id,
                item_id=item_id,
                cropped_image_path=cropped_filename,  # 只保存文件名
                original_category=category_name
            )
            return True, segmentation if segmentation else []
        except:
            pass
        return False, []
    
    # 构建文本提示（使用类别的多语言/同义词）
    prompts = CATEGORY_PROMPTS.get(category_name, [category_name])
    text_prompt = ". ".join(prompts) + "."
    
    # 临时保存图像以使用load_image
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        cv2.imwrite(tmp_path, image_source)
    
    try:
        # 使用Grounded-SAM-2处理
        refined_mask = process_image(tmp_path, text_prompt)
        
        # 检查mask是否有效
        if refined_mask.sum() == 0:
            return False, []
        
        # 保存mask图像（与原脚本一致）
        mask_filename = f"{image_id}_{item_id}_mask.png"
        mask_output_path = os.path.join(OUTPUT_MASKS_DIR, mask_filename)
        cv2.imwrite(mask_output_path, refined_mask * 255)
        
        # 创建带透明背景的图像（与原脚本一致）
        background_removed = cv2.cvtColor(image_source, cv2.COLOR_BGR2BGRA)
        background_removed[:, :, 3] = refined_mask * 255  # 设置alpha通道
        
        # 将透明区域的RGB通道也设置为255（白色）
        # 避免转换为RGB格式时背景被恢复
        transparent_mask = refined_mask == 0
        background_removed[transparent_mask, 0:3] = 255  # BGR通道设为白色
        
        # 用alpha通道做轮廓查找，裁剪到紧密边界（与原脚本一致）
        alpha_mask = background_removed[:, :, 3]
        contours, _ = cv2.findContours(alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 合并所有外部轮廓的边界
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # 转换为整数索引
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        
        # 计算正方形bbox（避免拉伸变形）
        w, h = x_max - x_min, y_max - y_min
        side = max(w, h)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        square_x1 = center_x - side // 2
        square_y1 = center_y - side // 2
        square_x2 = square_x1 + side
        square_y2 = square_y1 + side
        
        # 处理边界情况（可能超出图像）
        img_h, img_w = background_removed.shape[:2]
        if square_x1 >= 0 and square_y1 >= 0 and square_x2 <= img_w and square_y2 <= img_h:
            # 完全在图像内，直接裁剪
            square_img = background_removed[square_y1:square_y2, square_x1:square_x2]
        else:
            # 部分超出，需要padding（使用透明背景）
            square_img = np.full((side, side, 4), [255, 255, 255, 0], dtype=np.uint8)  # 白色透明背景
            valid_x1 = max(0, square_x1)
            valid_y1 = max(0, square_y1)
            valid_x2 = min(img_w, square_x2)
            valid_y2 = min(img_h, square_y2)
            dst_x1 = valid_x1 - square_x1
            dst_y1 = valid_y1 - square_y1
            dst_x2 = dst_x1 + (valid_x2 - valid_x1)
            dst_y2 = dst_y1 + (valid_y2 - valid_y1)
            square_img[dst_y1:dst_y2, dst_x1:dst_x2] = background_removed[valid_y1:valid_y2, valid_x1:valid_x2]
        
        # 缩放到 (224, 224)
        background_removed = cv2.resize(square_img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        
        # 保存裁剪后的图像（与原脚本一致）
        cv2.imwrite(cropped_output_path, background_removed)
        
        # 转换为多边形
        refined_polygons = mask_to_polygon(refined_mask)
        
        # 插入数据库记录（只保存文件名）
        insert_refined_item(
            db_path=db_path,
            image_id=image_id,
            item_id=item_id,
            cropped_image_path=cropped_filename,  # 只保存文件名
            original_category=category_name
        )
        
        return True, refined_polygons
    
    finally:
        # 删除临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_single_image(
    image_path: str,
    anno_path: str,
    output_path: str,
    image_id: str,
    db_path: str
) -> Tuple[bool, str, Dict]:
    """处理单张图片的所有items"""
    try:
        # 加载标注
        anno = load_annotation(anno_path)
        source = anno.get("source", "")
        
        # 检查是否有符合条件的item
        items_to_process = []
        for key, item in anno.items():
            if key.startswith("item") and isinstance(item, dict):
                if should_process_item(item, source):
                    items_to_process.append((key, item))
        
        if not items_to_process:
            return True, "No items match filter criteria", {}
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            return False, f"Failed to load image: {image_path}", {}
        
        # 处理每个item
        stats = {"total": len(items_to_process), "success": 0, "failed": 0}
        
        for key, item in items_to_process:
            success, refined_seg = process_single_item(image, item, image_id, key, db_path)
            
            if success and refined_seg:
                anno[key]["segmentation"] = refined_seg
                stats["success"] += 1
            else:
                stats["failed"] += 1
        
        # 保存精细化的标注
        save_annotation(output_path, anno)
        
        return True, "Success", stats
    
    except Exception as e:
        return False, f"Error: {str(e)}\n{traceback.format_exc()}", {}


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="DeepFashion2 蒙版精细化 V3 (Grounded-SAM-2)")
    parser.add_argument("--max_count", type=int, default=0,
                       help="最大处理数量（0=全部）")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CROPPED_DIR, exist_ok=True)
    
    # 初始化数据库
    init_database(SQLITE_DB_PATH)
    
    print("=" * 60)
    print("DeepFashion2 蒙版精细化 V3 (Grounded-SAM-2)")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"SAM2模型: {SAM2_CHECKPOINT}")
    print(f"Grounding DINO模型: {GROUNDING_DINO_CHECKPOINT}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"Mask输出: {OUTPUT_MASKS_DIR}")
    print(f"裁剪图片输出: {OUTPUT_CROPPED_DIR}")
    print(f"SQLite数据库: {SQLITE_DB_PATH}")
    print(f"裁剪尺寸: {SIZE}x{SIZE}")
    print("=" * 60)
    
    # 获取所有标注文件
    anno_files = sorted(Path(TRAIN_ANNOS_DIR).glob("*.json"))
    print(f"找到 {len(anno_files)} 个标注文件")
    
    if args.max_count > 0:
        anno_files = anno_files[:args.max_count]
        print(f"限制处理数量: {args.max_count}")
    
    # 统计信息
    processed_count = 0
    success_count = 0
    skip_count = 0
    error_count = 0
    total_items = 0
    success_items = 0
    failed_items = 0
    
    # 处理每个文件
    try:
        for anno_path in tqdm(anno_files, desc="处理图片"):
            image_name = anno_path.stem + ".jpg"
            image_id = anno_path.stem
            image_path = os.path.join(TRAIN_IMAGE_DIR, image_name)
            
            # 检查图像是否存在
            if not os.path.exists(image_path):
                skip_count += 1
                continue
            
            # 输出路径
            output_path = os.path.join(OUTPUT_DIR, anno_path.name)
            
            # 处理图片
            success, message, stats = process_single_image(
                image_path, str(anno_path), output_path, image_id, SQLITE_DB_PATH
            )
            
            processed_count += 1
            
            if success:
                if "No items match" in message:
                    skip_count += 1
                else:
                    success_count += 1
                    total_items += stats.get("total", 0)
                    success_items += stats.get("success", 0)
                    failed_items += stats.get("failed", 0)
            else:
                error_count += 1
                print(f"\n处理失败: {anno_path.name}")
                print(f"  {message}")
    
    except KeyboardInterrupt:
        print("\n处理被用户中断")
    except Exception as e:
        stacktrace = traceback.format_exc()
        print("\n发生错误:")
        print(stacktrace)
    finally:
        # 释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总共处理: {processed_count}")
    print(f"成功精细化: {success_count}")
    print(f"跳过（不符合条件）: {skip_count}")
    print(f"错误: {error_count}")
    print(f"总items: {total_items}")
    print(f"成功items: {success_items}")
    print(f"失败items: {failed_items}")
    print("=" * 60)


if __name__ == "__main__":
    main()
