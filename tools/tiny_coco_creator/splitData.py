import json
import os
import shutil
from pycocotools.coco import COCO
import random

# 你的标注文件
ann_file = "tiny_coco_1k.json"
coco = COCO(ann_file)

# 你本地COCO2017图片路径（请改成你自己的路径）
coco_img_dir = "../coco/images/train2017"  # 或 val2017

# 新目录，存放划分后的图片
train_img_dir = "../tiny_coco/images/train_images"
val_img_dir = "../tiny_coco/images/val_images"
os.makedirs(train_img_dir, exist_ok = True)
os.makedirs(val_img_dir, exist_ok = True)

# 抽取所有图片id，随机打乱
img_ids = coco.getImgIds()
random.shuffle(img_ids)

# 80%训练，20%验证
split_idx = int(len(img_ids) * 0.8)
train_ids = img_ids[:split_idx]
val_ids = img_ids[split_idx:]


def copy_images(coco, img_ids, src_dir, dst_dir):
    for img in coco.loadImgs(img_ids):
        filename = img['file_name']
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"图片不存在: {src_path}")


# 拷贝训练图片
copy_images(coco, train_ids, coco_img_dir, train_img_dir)
# 拷贝验证图片
copy_images(coco, val_ids, coco_img_dir, val_img_dir)


# 生成划分后的COCO格式JSON
def filter_coco_by_imgids(coco, img_ids):
    imgs = coco.loadImgs(img_ids)
    ann_ids = coco.getAnnIds(imgIds = img_ids)
    anns = coco.loadAnns(ann_ids)
    cats = coco.loadCats(coco.getCatIds())
    return {
        "images": imgs,
        "annotations": anns,
        "categories": cats
    }


train_data = filter_coco_by_imgids(coco, train_ids)
val_data = filter_coco_by_imgids(coco, val_ids)

with open("../../pytorch-retinanet/tiny_coco/annotations/instances_train2017.json", "w") as f:
    json.dump(train_data, f)

with open("../../pytorch-retinanet/tiny_coco/annotations/instances_val2017.json", "w") as f:
    json.dump(val_data, f)
