import numpy as np
from tensorflow.keras.layers import Input, Lambda
from networks.darknet.darknet import Darknet
from networks.detection_network import DetectionNetwork
from networks.yolo.util.labels_loader import LabelsLoader
from networks.yolo.layers import YoloConv, YoloOutput
from networks.yolo.util.utils import yolo_boxes, yolo_nms
from tensorflow.keras import Model
from typing import List


class YoloNetwork(DetectionNetwork):
    __yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                               (59, 119), (116, 90), (156, 198), (373, 326)],
                              np.float32) / 416
    __yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    def __init__(self, lang: str):
        super().__init__(lang)
        self._model.load_weights('./networks/yolo/yolov3.tf')

    def get_model(self):
        return self._model

    @staticmethod
    def _load_labels(lang: str) -> List[str]:
        return LabelsLoader.load(lang)

    @staticmethod
    def _make_model(size=None, channels=3, anchors=__yolo_anchors, masks=__yolo_anchor_masks, classes=80):
        x = inputs = Input([size, size, channels], name='input')

        x_36, x_61, x = Darknet(name='yolo_darknet')(x)

        x = YoloConv(512, name='yolo_conv_0')(x)
        output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

        x = YoloConv(256, name='yolo_conv_1')((x, x_61))
        output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

        x = YoloConv(128, name='yolo_conv_2')((x, x_36))
        output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

        boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
        boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)

        outputs = Lambda(lambda x: yolo_nms(x, classes), name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return Model(inputs, outputs, name='yolov3')
