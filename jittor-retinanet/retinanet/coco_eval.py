import jittor
from pycocotools.cocoeval import COCOeval
import json
# import torch
import jittor as jt

def evaluate_coco(dataset, model, threshold = 0.05):
    model.eval()

    # with torch.no_grad():
    with ((jt.no_grad())):

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            #  (C, H, W)
            # print(data['img'].shape)
            data['img'] = jittor.permute(data['img'], [2, 0, 1])
            # print(data['img'] = data['img'].permute(data['img'], [2, 0, 1]))
            # print(data['img'].shape)
            # print(model(data['img'].float32().unsqueeze(0)))
            scores, labels, boxes = model(data['img'].float32().unsqueeze(0))[0]

            # result =
            # scores, labels, boxes
            # output = model(data['img'].float32().unsqueeze(0))
            # print(type(output))  # 是 list？tuple？tensor？
            # print(len(output))  # 如果是 list，看看有几个元素
            # print(output)
            # scores = result[0]
            # labels = result[1]
            # boxes = result[2]



            # run network
            # if torch.cuda.is_available():
            # scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim = 0))
            # else:
            #     scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim = 0))

            # scores = scores.cpu()
            # labels = labels.cpu()
            # boxes = boxes.cpu()

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

# x = torch.tensor([[1, 2], [3, 4], [5, 6]])
# x
# print(x.shape)
# x.unsqueeze(dim=0)
# print(x.shape)
# from pycocotools.cocoeval import COCOeval
# import json
# import jittor as jt
# import os # 导入 os 模块用于路径操作和文件清理
#
# # 假设 jt.flags.use_cuda = 1 已经在程序的入口处设置
#
# def evaluate_coco(dataloader, model, threshold=0.05, output_dir='eval_results'): # 添加 output_dir 参数
#     model.eval()
#
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#     results_filepath = os.path.join(output_dir, f'{dataloader.set_name}_bbox_results.json')
#
#     with jt.no_grad():
#         results = []
#         image_ids = []
#
#         for index, data in enumerate(dataloader):#range(len(dataloader)):
#             # data = dataloader[index]
#             scale = data['scale']
#
#             # 假设 data['img'] 已经经过 collater 处理为 (C, H, W) 浮点张量
#             # 如果不是，则保持原样：model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
#             # scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim = 0))
#
#             print(model(data['img']).unsqueeze(dim=0))
#             print(model(data['img']).unsqueeze(dim=0).shape)
#
#             scores, labels, boxes = model(data['img']).unsqueeze(dim=0)
#
#             # 将结果移到CPU进行后续处理（Pycocotools通常需要Python列表或Numpy数组）
#             scores = scores.cpu()
#             labels = labels.cpu()
#             boxes = boxes.cpu()
#
#             boxes /= scale
#
#             if boxes.shape[0] > 0:
#                 boxes[:, 2] -= boxes[:, 0]
#                 boxes[:, 3] -= boxes[:, 1]
#
#                 # 优化：直接过滤，而不是依赖排序后的break
#                 for box_id in range(boxes.shape[0]):
#                     score = float(scores[box_id])
#                     label = int(labels[box_id])
#                     box = boxes[box_id, :]
#
#                     if score < threshold:
#                         continue # 修改为 continue，确保处理所有高于阈值的框
#
#                     image_result = {
#                         'image_id': dataloader.image_ids[index],
#                         'category_id': dataloader.label_to_coco_label(label),
#                         'score': score, # score 已经是 float
#                         'bbox': box.tolist(),
#                     }
#                     results.append(image_result)
#
#             image_ids.append(dataloader.image_ids[index])
#             # 使用 sys.stdout.write 避免每次 print 换行，确保进度条在一行
#             import sys
#             sys.stdout.write(f'\rEvaluating: {index+1}/{len(dataloader)}')
#             sys.stdout.flush()
#         sys.stdout.write('\n') # 评估完成后换行
#
#         if not len(results):
#             print(f"No detections found above threshold {threshold} for {dataloader.set_name}.")
#             model.train() # 恢复训练模式
#             return 0.0 # 返回 0.0 表示mAP为0
#
#     # write output
#     try:
#         with open(results_filepath, 'w') as f:
#             json.dump(results, f, indent = 4)
#     except IOError as e:
#         print(f"Error writing results to file {results_filepath}: {e}")
#         model.train()
#         return 0.0
#
#     # load results in COCO evaluation tool
#     coco_true = dataloader.coco
#     try:
#         coco_pred = coco_true.loadRes(results_filepath)
#     except Exception as e:
#         print(f"Error loading prediction results from {results_filepath}: {e}")
#         model.train()
#         return 0.0
#
#
#     # run COCO evaluation
#     coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
#     coco_eval.params.imgIds = image_ids
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()
#
#     model.train() # 评估完成后，将模型设置回训练模式
#
#     # 清理临时文件
#     try:
#         os.remove(results_filepath)
#     except OSError as e:
#         print(f"Error removing temporary results file {results_filepath}: {e}")
#
#     return coco_eval.stats[0] # 返回主指标 mAP (IoU=0.50:0.95)
