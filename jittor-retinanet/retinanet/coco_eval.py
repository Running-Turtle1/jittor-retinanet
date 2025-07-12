import jittor
from pycocotools.cocoeval import COCOeval
import json
# import torch
import jittor as jt

def evaluate_coco(dataset, model, threshold = 0.05):
    model.eval()

    # with torch.no_grad():
    with (jt.no_grad()):

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            #  (C, H, W)
            data['img'] = jittor.permute(data['img'], [2, 0, 1])
            scores, labels, boxes = model(data['img'].float32().unsqueeze(0))[0]
            boxes /= scale

            if boxes.shape[0] > 0:
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    if score < threshold:
                        break

                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }
                    results.append(image_result)

            image_ids.append(dataset.image_ids[index])
            print(f'{index}/{len(dataset)}', end = '\r')

        if not len(results):
            model.train()
            return 0.0

    # write output
    with open(f'{dataset.set_name}_bbox_results.json', 'w') as f:
        json.dump(results, f, indent = 4)

    # load results in COCO evaluation tool
    coco_true = dataset.coco
    coco_pred = coco_true.loadRes(f'{dataset.set_name}_bbox_results.json')

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    model.train()


    return coco_eval.stats[0]  # 返回主指标 mAP (IoU=0.50:0.95)

