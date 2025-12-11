import json
import os
import shutil
import random
from PIL import Image

# ================================
# 你的数据集路径（已替换）
# ================================
IMAGE_DIR = r"D:\python\Cotton Weed Detect\dataprocess\cottonweed_split\train\images"
ANNOT_PATH = r"D:\python\Cotton Weed Detect\dataprocess\cottonweed_split\train\annotations"
OUTPUT_ROOT = r"D:\python\Cotton Weed Detect\dataprocess\cottonweed_split\train\noisy datasets"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 三个噪声比例
NOISE_RATIOS = [0.05, 0.10, 0.20]

# CottonWeed 为 3 类
NUM_CLASSES = 3

# 类别名称到 ID 的映射
CLASS_NAME_TO_ID = {
    "carpetweed": 0,
    "morningglory": 1,
    "palmer_amaranth": 2
}

# ID 到类别名称的映射
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


# ================================
# VIA 格式转换函数
# ================================
def via_to_coco(via_data, image_path):
    """
    将 VIA 格式转换为 COCO 格式
    
    Args:
        via_data: VIA 格式的 JSON 数据
        image_path: 图像文件路径（用于获取尺寸）
    
    Returns:
        COCO 格式的字典
    """
    # 获取图像尺寸
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"警告: 无法读取图像 {image_path}: {e}")
        return None
    
    # 获取 VIA 数据中的第一个键（通常是 via_<filename>）
    via_key = list(via_data.keys())[0]
    via_entry = via_data[via_key]
    
    filename = via_entry["filename"]
    regions = via_entry.get("regions", [])
    
    # 创建 COCO 格式
    coco = {
        "images": [{
            "id": 1,
            "file_name": filename,
            "width": img_w,
            "height": img_h
        }],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "carpetweed", "supercategory": "weed"},
            {"id": 1, "name": "morningglory", "supercategory": "weed"},
            {"id": 2, "name": "palmer_amaranth", "supercategory": "weed"}
        ]
    }
    
    # 转换标注
    for idx, region in enumerate(regions):
        shape_attrs = region.get("shape_attributes", {})
        region_attrs = region.get("region_attributes", {})
        
        if shape_attrs.get("name") != "rect":
            continue
        
        x = shape_attrs.get("x", 0)
        y = shape_attrs.get("y", 0)
        w = shape_attrs.get("width", 0)
        h = shape_attrs.get("height", 0)
        
        class_name = region_attrs.get("class", "")
        category_id = CLASS_NAME_TO_ID.get(class_name, 0)
        
        coco["annotations"].append({
            "id": idx + 1,
            "image_id": 1,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        })
    
    return coco


def coco_to_via(coco_data, original_via_key=None):
    """
    将 COCO 格式转换回 VIA 格式
    
    Args:
        coco_data: COCO 格式的字典
        original_via_key: 原始的 VIA 键名（如果为 None，则从文件名生成）
    
    Returns:
        VIA 格式的字典
    """
    if len(coco_data["images"]) == 0:
        return None
    
    img_info = coco_data["images"][0]
    filename = img_info["file_name"]
    
    # 生成 VIA 键名
    if original_via_key is None:
        via_key = f"via_{filename.replace('.jpg', '').replace('.png', '')}"
    else:
        via_key = original_via_key
    
    # 转换标注
    regions = []
    for ann in coco_data["annotations"]:
        x, y, w, h = ann["bbox"]
        category_id = ann["category_id"]
        class_name = CLASS_ID_TO_NAME.get(category_id, "carpetweed")
        
        regions.append({
            "shape_attributes": {
                "name": "rect",
                "x": x,
                "y": y,
                "width": w,
                "height": h
            },
            "region_attributes": {
                "class": class_name
            }
        })
    
    via_data = {
        via_key: {
            "filename": filename,
            "regions": regions,
            "size": -1,
            "file_attributes": []
        }
    }
    
    return via_data


# ================================
# 读取 COCO JSON
# ================================
def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================
# 写回 COCO JSON
# ================================
def save_coco(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ================================
# 读取 VIA JSON
# ================================
def load_via(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================
# 写回 VIA JSON
# ================================
def save_via(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ================================
# 1) Missing（漏标）
# ================================
def apply_missing(coco, annotations, ratio):
    return [ann for ann in annotations if random.random() > ratio]


# ================================
# 2) Spurious（加假框）
# ================================
def apply_spurious(coco, annotations, ratio):
    images = {img["id"]: img for img in coco["images"]}
    anns = annotations.copy()

    # 获取现有标注的最大 ID，避免冲突
    max_id = max([ann["id"] for ann in annotations], default=0)
    next_id = max_id + 1

    num_fake = int(len(annotations) * ratio)

    for _ in range(num_fake):
        img_info = random.choice(coco["images"])
        img_w = img_info["width"]
        img_h = img_info["height"]

        w = random.uniform(0.05, 0.2) * img_w
        h = random.uniform(0.05, 0.2) * img_h
        x = random.uniform(0, img_w - w)
        y = random.uniform(0, img_h - h)

        fake_ann = {
            "id": next_id,
            "image_id": img_info["id"],
            "category_id": random.randint(0, NUM_CLASSES - 1),
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0
        }
        anns.append(fake_ann)
        next_id += 1

    return anns


# ================================
# 3) Mislocated（框位置移动）
# ================================
def apply_mislocated(coco, annotations, ratio):
    images = {img["id"]: img for img in coco["images"]}
    anns = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        
        # 获取对应图像的尺寸
        img_info = images.get(ann["image_id"])
        if img_info is None:
            # 如果找不到图像信息，保持原标注不变
            anns.append(ann)
            continue
        
        img_w = img_info["width"]
        img_h = img_info["height"]

        shift = ratio * 0.5

        def shift_val(v):
            return max(0, v + random.uniform(-shift, shift) * v)

        x_new = shift_val(x)
        y_new = shift_val(y)
        
        # 确保边界框不超出图像范围
        x_new = max(0, min(x_new, img_w - w))
        y_new = max(0, min(y_new, img_h - h))
        
        # 确保宽度和高度不会导致超出边界
        if x_new + w > img_w:
            w = img_w - x_new
        if y_new + h > img_h:
            h = img_h - y_new

        anns.append({
            **ann,
            "bbox": [x_new, y_new, w, h],
            "area": w * h  # 更新面积
        })
    return anns


# ================================
# 4) Mislabeled（错误类别）
# ================================
def apply_mislabeled(coco, annotations, ratio):
    anns = []
    for ann in annotations:
        if random.random() < ratio:
            new_c = random.choice([c for c in range(NUM_CLASSES) if c != ann["category_id"]])
        else:
            new_c = ann["category_id"]

        anns.append({
            **ann,
            "category_id": new_c
        })
    return anns


# ================================
# 总控函数（生成 12 套数据）
# ================================
def generate_noise_sets():
    for annot_file in os.listdir(ANNOT_PATH):
        if not annot_file.endswith(".json"):
            continue

        annot_path = os.path.join(ANNOT_PATH, annot_file)
        
        # 读取 VIA 格式
        via_data = load_via(annot_path)
        
        # 获取对应的图像路径
        base_name = os.path.splitext(annot_file)[0]
        # 尝试不同的图像扩展名
        img_extensions = [".jpg", ".jpeg", ".png"]
        img_path = None
        for ext in img_extensions:
            potential_path = os.path.join(IMAGE_DIR, base_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            print(f"警告: 找不到图像文件 {base_name}，跳过")
            continue
        
        # 转换为 COCO 格式
        coco = via_to_coco(via_data, img_path)
        if coco is None:
            print(f"警告: 无法转换 {annot_file}，跳过")
            continue
        
        # 保存原始的 VIA 键名
        original_via_key = list(via_data.keys())[0]

        for ratio in NOISE_RATIOS:
            percent = int(ratio * 100)

            # 逐类生成
            tasks = {
                "missing": apply_missing,
                "spurious": apply_spurious,
                "mislocated": apply_mislocated,
                "mislabeled": apply_mislabeled
            }

            for noise_name, func in tasks.items():
                out_dir = os.path.join(OUTPUT_ROOT, f"{noise_name}_{percent}")
                img_out = os.path.join(out_dir, "images")
                ann_out = os.path.join(out_dir, "annotations")

                os.makedirs(img_out, exist_ok=True)
                os.makedirs(ann_out, exist_ok=True)

                # 复制对应的图像（只复制当前处理的图像）
                if os.path.exists(img_path):
                    img_filename = os.path.basename(img_path)
                    shutil.copy(img_path, os.path.join(img_out, img_filename))

                # 处理 annotations
                anns_new = func(coco, coco["annotations"], ratio)
                coco_new = coco.copy()
                coco_new["annotations"] = anns_new

                # 转换回 VIA 格式
                via_new = coco_to_via(coco_new, original_via_key)
                if via_new:
                    save_via(via_new, os.path.join(ann_out, annot_file))

                print(f"生成成功：{noise_name}_{percent} - {annot_file}")

    print("🎉🎉 全部 12 套噪声数据集生成完毕！")


generate_noise_sets()