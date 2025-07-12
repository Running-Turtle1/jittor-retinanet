import argparse
import warnings

import jittor as jt
# from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval
from jittor import transform, optim
# assert torch.__version__.split('.')[0] == '1'

# print(f'CUDA available: {torch.cuda.is_available()}')

warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.autocast.*",
    category=FutureWarning
)

def main(args = None):
    # 程序的简要说明，当输入 python coco_validation.py 时, 会显示 description
    parser = argparse.ArgumentParser(description = 'Simple training script for training a RetinaNet network.')

    # 添加参数
    parser.add_argument('--coco_path', default = './coco', help = 'Path to COCO directory')
    parser.add_argument('--model_path', default = '/xzp_qzh/Jittor/workspace/file/jittor-retinanet/logs/tiny_coco_retinanet_epoch1.pt', help = 'Path to model', type = str)

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name = 'val2017',
                              transform = transform.Compose([Normalizer(), Resizer()]))

    # Create the model
    retinanet = model.resnet50(num_classes = dataset_val.num_classes(), pretrained = True)

    # use_gpu = True

    # if use_gpu:
    #     if torch.cuda.is_available():
    #         retinanet  # 把模型从 CPU 内存搬到 GPU 内存
    jt.flags.use_cuda = True

    # if torch.cuda.is_available():
    #     retinanet = torch.load(parser.model_path)  # 直接加载整个模型
    #     retinanet = torch.nn.DataParallel(retinanet)
    # else:
    #     retinanet = torch.load(parser.model_path)
    #     retinanet = torch.nn.DataParallel(retinanet)

    print(parser.model_path)
    # retinanet = jt.load(args.model_path)

    retinanet.load(parser.model_path)

    retinanet.training = False
    retinanet.eval()
    retinanet.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet)  # 调用 COCO 数据集的评估函数，输入验证集 dataset_val 和模型 retinanet


if __name__ == '__main__':
    main()
