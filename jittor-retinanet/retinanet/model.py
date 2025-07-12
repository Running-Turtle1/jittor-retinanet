import jittor.nn as nn
import jittor as jt
import math
from jittor import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses

import jittor.init as init


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size = 256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size = 1, stride = 1, padding = 0)
        self.P5_upsampled = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, stride = 1, padding = 1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size = 1, stride = 1, padding = 0)
        self.P4_upsampled = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, stride = 1, padding = 1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size = 1, stride = 1, padding = 0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, stride = 1, padding = 1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size = 3, stride = 2, padding = 1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, stride = 2, padding = 1)

    def execute(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors = 9, feature_size = 256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size = 3, padding = 1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, padding = 1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, padding = 1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, padding = 1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size = 3, padding = 1)

    def execute(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors = 9, num_classes = 80, prior = 0.01, feature_size = 256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size = 3, padding = 1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, padding = 1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, padding = 1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size = 3, padding = 1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size = 3, padding = 1)
        self.output_act = nn.Sigmoid()

    def execute(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes = num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                std = math.sqrt(2. / n)
                m.weight.data = jt.normal(0, std, size = m.weight.data.shape)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.assign(jt.ones(m.weight.shape, dtype = m.weight.dtype))
                m.bias.assign(jt.zeros(m.bias.shape, dtype = m.bias.dtype))

        prior = 0.01

        # Initialize the classification and regression models
        self.classificationModel.output.weight.assign(jt.zeros(self.classificationModel.output.weight.shape,
                                                               dtype = self.classificationModel.output.weight.dtype))
        self.classificationModel.output.bias.assign(
            jt.full(self.classificationModel.output.bias.shape, -math.log((1.0 - prior) / prior),
                    dtype = self.classificationModel.output.bias.dtype))
        self.regressionModel.output.weight.assign(
            jt.zeros(self.regressionModel.output.weight.shape, dtype = self.regressionModel.output.weight.dtype))
        self.regressionModel.output.bias.assign(
            jt.zeros(self.regressionModel.output.bias.shape, dtype = self.regressionModel.output.bias.dtype))

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def execute(self, inputs):

        if self.is_training():
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        # print(img_batch)
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = jt.concat([self.regressionModel(feature) for feature in features], dim = 1)

        classification = jt.concat([self.classificationModel(feature) for feature in features], dim = 1)

        anchors = self.anchors(img_batch)

        if self.is_training():
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            # 推理模式
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            # 用于累积批次中每张图像的最终检测结果
            batch_detections = []

            batch_size = img_batch.shape[0]
            for b_idx in range(batch_size):
                # 获取当前图像的分类预测和回归框
                # classification: (batch_size, num_total_anchors, num_classes)
                # regression: (batch_size, num_total_anchors, 4)
                # transformed_anchors: (batch_size, num_total_anchors, 4)

                # 当前图像的分类预测，形状为 (num_total_anchors, num_classes)
                current_img_classification = classification[b_idx, :, :]
                # 当前图像的回归框，形状为 (num_total_anchors, 4)
                current_img_transformed_anchors = transformed_anchors[b_idx, :, :]

                # 用于累积当前图像的最终检测结果
                finalScores_img_list = []
                finalAnchorBoxesIndexes_img_list = []
                finalAnchorBoxesCoordinates_img_list = []

                # 遍历每个类别
                for class_idx in range(current_img_classification.shape[1]):
                    # 获取当前类别在当前图像上的分数
                    # scores 形状为 (num_total_anchors,)
                    scores = current_img_classification[:, class_idx]

                    # 筛选出分数高于阈值的检测 (0.05)
                    scores_over_thresh = (scores > 0.05)

                    # 如果没有高于阈值的检测，则跳过
                    if scores_over_thresh.sum() == 0:
                        continue

                    # 根据阈值筛选分数和对应的锚框
                    scores_filtered = scores[scores_over_thresh]
                    anchorBoxes_filtered = current_img_transformed_anchors[scores_over_thresh]

                    # 如果筛选后没有有效的检测，则跳过
                    if scores_filtered.numel() == 0:
                        continue

                    # 执行NMS
                    # nms函数期望输入是 (N, 4) 的boxes和 (N,) 的scores
                    # 建议将 NMS IOU 阈值从 0.05 提高到 0.5 或 0.6，以减少重复检测
                    nms_iou_threshold = 0.5 # 修正建议：将此值从 0.05 更改为 0.5 或更高
                    # anchors_nms_idx = nms(anchorBoxes_filtered, scores_filtered, nms_iou_threshold)
                    dets = jt.concat([anchorBoxes_filtered, scores_filtered.unsqueeze(-1)], dim = 1)
                    anchors_nms_idx = nms(dets, nms_iou_threshold)

                    # 累积当前图像、当前类别的NMS结果
                    finalScores_img_list.append(scores_filtered[anchors_nms_idx])
                    finalAnchorBoxesIndexes_img_list.append(
                        jt.array([class_idx] * anchors_nms_idx.shape[0], dtype = jt.int32))
                    finalAnchorBoxesCoordinates_img_list.append(anchorBoxes_filtered[anchors_nms_idx])

                # 对当前图像的所有类别结果进行最终的合并
                if len(finalScores_img_list) > 0:
                    finalScores_img = jt.concat(finalScores_img_list)
                    finalAnchorBoxesIndexes_img = jt.concat(finalAnchorBoxesIndexes_img_list)
                    finalAnchorBoxesCoordinates_img = jt.concat(finalAnchorBoxesCoordinates_img_list)
                else:
                    # 如果当前图像没有检测结果，创建正确形状的空张量
                    finalScores_img = jt.array([], dtype = jt.float32)
                    finalAnchorBoxesIndexes_img = jt.array([], dtype = jt.int32)
                    finalAnchorBoxesCoordinates_img = jt.array([], dtype = jt.float32).reshape(0, 4)

                # 将当前图像的结果添加到批次总结果中
                batch_detections.append([finalScores_img, finalAnchorBoxesIndexes_img, finalAnchorBoxesCoordinates_img])

            # 返回所有图像的最终检测结果列表
            return batch_detections


def resnet50(num_classes, pretrained = False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load("jittorhub://resnet50.pkl")
    return model


if __name__ == '__main__':
    print(jt.full((3, 3), 1, dtype = jt.float32))

    print(jt.array([[1, 2, 3], [4, 5, 6]], dtype = jt.float32))

    # Unit tests
    print("\n--- Running Unit Tests ---")

    # Test PyramidFeatures
    print("Testing PyramidFeatures...")
    batch_size = 2  # Increased batch size for better testing of batch processing
    C3_channels = 512  # Example output channels from ResNet layer2 (Bottleneck expansion 4 * 128)
    C4_channels = 1024  # Example output channels from ResNet layer3 (Bottleneck expansion 4 * 256)
    C5_channels = 2048  # Example output channels from ResNet layer4 (Bottleneck expansion 4 * 512)
    feature_size = 256

    # Simulate feature map sizes after ResNet layers (assuming input image is 512x512)
    # Layer 2 (C3): input_res / 8, e.g., 512/8 = 64
    # Layer 3 (C4): input_res / 16, e.g., 512/16 = 32
    # Layer 4 (C5): input_res / 32, e.g., 512/32 = 16
    C3 = jt.rand(batch_size, C3_channels, 64, 64)
    C4 = jt.rand(batch_size, C4_channels, 32, 32)
    C5 = jt.rand(batch_size, C5_channels, 16, 16)

    pyramid_features_model = PyramidFeatures(C3_channels, C4_channels, C5_channels, feature_size)
    P3, P4, P5, P6, P7 = pyramid_features_model([C3, C4, C5])

    # Expected output shapes
    # P5: C5 (B, 2048, 16, 16) -> P5_1 (B, 256, 16, 16) -> P5_2 (B, 256, 16, 16)
    # P4: C4 (B, 1024, 32, 32) + P5_upsampled (B, 256, 32, 32) -> P4_2 (B, 256, 32, 32)
    # P3: C3 (B, 512, 64, 64) + P4_upsampled (B, 256, 64, 64) -> P3_2 (B, 256, 64, 64)
    # P6: C5 (B, 2048, 16, 16) -> P6 (B, 256, 8, 8) (stride 2)
    # P7: P6 (B, 256, 8, 8) -> P7_2 (B, 256, 4, 4) (stride 2)

    assert P3.shape == (batch_size, feature_size, 64, 64), f"P3 shape mismatch: {P3.shape}"
    assert P4.shape == (batch_size, feature_size, 32, 32), f"P4 shape mismatch: {P4.shape}"
    assert P5.shape == (batch_size, feature_size, 16, 16), f"P5 shape mismatch: {P5.shape}"
    assert P6.shape == (batch_size, feature_size, 8, 8), f"P6 shape mismatch: {P6.shape}"
    assert P7.shape == (batch_size, feature_size, 4, 4), f"P7 shape mismatch: {P7.shape}"
    print("PyramidFeatures test passed!")

    # Test RegressionModel
    print("Testing RegressionModel...")
    num_anchors = 9
    reg_model = RegressionModel(feature_size, num_anchors = num_anchors)
    # Input feature map for regression model (e.g., P3 from PyramidFeatures)
    reg_input = jt.rand(batch_size, feature_size, 64, 64)
    reg_output = reg_model(reg_input)
    # Expected output shape: B x (H*W*num_anchors) x 4
    expected_reg_output_shape = (batch_size, 64 * 64 * num_anchors, 4)
    assert reg_output.shape == expected_reg_output_shape, f"RegressionModel output shape mismatch: {reg_output.shape}, expected: {expected_reg_output_shape}"
    print("RegressionModel test passed!")

    # Test ClassificationModel
    print("Testing ClassificationModel...")
    num_classes = 80
    cls_model = ClassificationModel(feature_size, num_anchors = num_anchors, num_classes = num_classes)
    # Input feature map for classification model (e.g., P3 from PyramidFeatures)
    cls_input = jt.rand(batch_size, feature_size, 64, 64)
    cls_output = cls_model(cls_input)
    # Expected output shape: B x (H*W*num_anchors) x num_classes
    expected_cls_output_shape = (batch_size, 64 * 64 * num_anchors, num_classes)
    assert cls_output.shape == expected_cls_output_shape, f"ClassificationModel output shape mismatch: {cls_output.shape}, expected: {expected_cls_output_shape}"
    # Check if values are within [0, 1] due to Sigmoid
    assert jt.all(cls_output >= 0.0) and jt.all(cls_output <= 1.0), "ClassificationModel output values not in [0, 1]"
    print("ClassificationModel test passed!")

    # Test ResNet (forward pass in eval mode)
    print("Testing ResNet (evaluation mode)...")
    num_classes_resnet = 80
    resnet_model = resnet50(num_classes_resnet, pretrained = False)
    resnet_model.eval()  # Set to evaluation mode

    # Create dummy image batch (e.g., 2, 3, 512, 512)
    dummy_img_batch = jt.rand(2, 3, 512, 512)  # Use batch_size = 2 for testing
    batch_results = resnet_model(dummy_img_batch) # Now returns a list of results per image

    assert isinstance(batch_results, list), "Output should be a list for batch inference"
    assert len(batch_results) == dummy_img_batch.shape[0], "Number of results should match batch size"

    for i, (scores, labels, boxes) in enumerate(batch_results):
        print(f"Image {i+1} results: scores shape {scores.shape}, labels shape {labels.shape}, boxes shape {boxes.shape}")
        assert scores.ndim == 1, f"Scores for image {i+1} should be 1D, got {scores.ndim}"
        assert labels.ndim == 1, f"Labels for image {i+1} should be 1D, got {labels.ndim}"
        assert boxes.ndim == 2 and boxes.shape[1] == 4, f"Boxes for image {i+1} should be 2D with last dim 4, got {boxes.shape}"
        assert scores.shape[0] == labels.shape[0] == boxes.shape[
            0], f"Scores, labels, and boxes for image {i+1} should have the same number of detections"
    print("ResNet evaluation mode test passed!")

    print("\nAll unit tests passed!")
