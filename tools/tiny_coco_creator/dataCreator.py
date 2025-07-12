from pycocotools.coco import COCO
import json, os, shutil

src_ann_file = '../coco/annotations/instances_train2017.json'
coco = COCO(src_ann_file)

img_ids = coco.getImgIds()[:1000]
imgs = coco.loadImgs(img_ids)

subset = {
    "images": imgs,
    "annotations": [],
    "categories": coco.loadCats(coco.getCatIds())
}

ann_ids = coco.getAnnIds(imgIds=img_ids)
subset["annotations"] = coco.loadAnns(ann_ids)

with open("tiny_coco_1k.json", "w") as f:
    json.dump(subset, f)