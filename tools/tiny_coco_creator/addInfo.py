import json
import os

# 替换为你的文件路径
json_path = '../../pytorch-retinanet/tiny_coco/annotations/instances_train2017.json'

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 如果没有 info 字段，就添加
if 'info' not in data:
    data['info'] = {
        "description": "Tiny COCO Dataset",
        "version": "1.0",
        "year": 2025,
        "contributor": "Wei Pengchao",
        "date_created": "2025-07-09"
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print("✅ 已添加 info 字段。")
else:
    print("ℹ️ 已存在 info 字段，无需修改。")
