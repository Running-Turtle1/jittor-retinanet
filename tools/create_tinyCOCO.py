import argparse
import json
import os
import random
import shutil

from pycocotools.coco import COCO


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_SRC_ANN = os.path.join(REPO_ROOT, 'coco', 'annotations', 'instances_train2017.json')
DEFAULT_SRC_IMG_DIR = os.path.join(REPO_ROOT, 'coco', 'images', 'train2017')
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, 'tiny_coco')
DEFAULT_SUBSET_JSON = os.path.join(SCRIPT_DIR, 'tiny_coco_1k.json')


def build_subset(coco_api, image_ids):
    return {
        'images': coco_api.loadImgs(image_ids),
        'annotations': coco_api.loadAnns(coco_api.getAnnIds(imgIds=image_ids)),
        'categories': coco_api.loadCats(coco_api.getCatIds()),
    }


def copy_images(coco_api, image_ids, src_dir, dst_dir):
    missing = 0
    for img in coco_api.loadImgs(image_ids):
        filename = img['file_name']
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            missing += 1
            print(f'Missing image: {src_path}')
    return missing


def add_info_field(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'info' not in data:
        data['info'] = {
            'description': 'Tiny COCO Dataset',
            'version': '1.0',
            'year': 2025,
            'contributor': 'Wei Pengchao',
            'date_created': '2025-07-09',
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Create a tiny COCO-style dataset from COCO train2017.')
    parser.add_argument('--src_ann', default=DEFAULT_SRC_ANN, help='Path to the source COCO annotation JSON')
    parser.add_argument('--src_img_dir', default=DEFAULT_SRC_IMG_DIR, help='Path to the source COCO image directory')
    parser.add_argument('--out_root', default=DEFAULT_OUT_ROOT, help='Output directory for the tiny COCO dataset')
    parser.add_argument('--subset_json', default=DEFAULT_SUBSET_JSON, help='Path to save the intermediate subset JSON')
    parser.add_argument('--num_images', type=int, default=1000, help='Number of source images to keep')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio, between 0 and 1')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the train/val split')
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError('--train_ratio must be between 0 and 1')

    train_img_dir = os.path.join(args.out_root, 'images', 'train2017')
    val_img_dir = os.path.join(args.out_root, 'images', 'val2017')
    annotation_dir = os.path.join(args.out_root, 'annotations')
    train_ann_file = os.path.join(annotation_dir, 'instances_train2017.json')
    val_ann_file = os.path.join(annotation_dir, 'instances_val2017.json')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    coco = COCO(args.src_ann)
    image_ids = coco.getImgIds()[:args.num_images]

    subset = build_subset(coco, image_ids)
    with open(args.subset_json, 'w', encoding='utf-8') as f:
        json.dump(subset, f)

    rng = random.Random(args.seed)
    rng.shuffle(image_ids)
    split_idx = int(len(image_ids) * args.train_ratio)
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]

    missing_train = copy_images(coco, train_ids, args.src_img_dir, train_img_dir)
    missing_val = copy_images(coco, val_ids, args.src_img_dir, val_img_dir)

    with open(train_ann_file, 'w', encoding='utf-8') as f:
        json.dump(build_subset(coco, train_ids), f)

    with open(val_ann_file, 'w', encoding='utf-8') as f:
        json.dump(build_subset(coco, val_ids), f)

    add_info_field(train_ann_file)
    add_info_field(val_ann_file)

    print(f'Wrote subset JSON: {args.subset_json}')
    print(f'Wrote tiny COCO dataset: {args.out_root}')
    print(f'Train images: {len(train_ids)}, Val images: {len(val_ids)}')
    print(f'Missing copied images: train={missing_train}, val={missing_val}')


if __name__ == '__main__':
    main()
