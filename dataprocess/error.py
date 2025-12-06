import json
import os
import shutil
import random

# ================================
# 你的数据集路径（已替换）
# ================================
IMAGE_DIR = r"C:\Users\shish\Desktop\cottonweed_split\train\images"
ANNOT_PATH = r"C:\Users\shish\Desktop\cottonweed_split\train\annotations"
OUTPUT_ROOT = r"C:\Users\shish\Desktop\cottonweed_split\noise_datasets"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 三个噪声比例
NOISE_RATIOS = [0.05, 0.10, 0.20]

# CottonWeed 为 3 类
NUM_CLASSES = 3


# ================================
# 读取 COCO JSON
# ================================
def load_coco(path):
    with open(path, "r") as f:
        return json.load(f)


# ================================
# 写回 COCO JSON
# ================================
def save_coco(data, path):
    with open(path, "w") as f:
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
            "category_id": random.randint(1, NUM_CLASSES),
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
            new_c = random.choice([c for c in range(1, NUM_CLASSES + 1) if c != ann["category_id"]])
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

        coco = load_coco(os.path.join(ANNOT_PATH, annot_file))

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

                # 复制图像
                for img in os.listdir(IMAGE_DIR):
                    shutil.copy(os.path.join(IMAGE_DIR, img),
                                os.path.join(img_out, img))

                # 处理 annotations
                anns_new = func(coco, coco["annotations"], ratio)
                coco_new = coco.copy()
                coco_new["annotations"] = anns_new

                save_coco(coco_new, os.path.join(ann_out, annot_file))

                print(f"生成成功：{noise_name}_{percent}")

    print("🎉🎉 全部 12 套噪声数据集生成完毕！")


generate_noise_sets()