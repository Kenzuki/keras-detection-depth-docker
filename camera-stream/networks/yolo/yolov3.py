from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw
from numpy import expand_dims
from networks.yolo.util.utils import decode_netout, correct_yolo_boxes, do_nms
from networks.yolo.util.labels_loader import LabelsLoader
from networks.yolo.util.bounding_box import BoundingBox
from typing import List
import pyrealsense2 as rs


class YoloNetwork:
    __anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    __threshold = 0.6

    def __init__(self, lang='en') -> None:
        self.__model = load_model('./networks/yolo/model.h5', compile=False)
        self.__labels = LabelsLoader.load(lang)

    def predict(self, img: Image, depth: rs.depth_frame):
        image, img_w, img_h = self.__preprocess_image(img)
        yhat = self.__model.predict(image)
        boxes = list()
        for i in range(len(yhat)):
            boxes += decode_netout(yhat[i][0], self.__anchors[i], self.__threshold, 416, 416)
        correct_yolo_boxes(boxes, img_h, img_w, 416, 416)
        v_boxes, v_labels, v_scores = self.__get_boxes(boxes)
        selected_boxes, selected_labels, selected_scores = do_nms(v_boxes, v_labels, v_scores, 0.5)
        for i in range(len(selected_boxes)):
            print(selected_labels[i], selected_scores[i])
        return self.__draw_prediction(img, selected_boxes, selected_labels, selected_scores, depth)

    def __get_boxes(self, boxes: List[BoundingBox]):
        v_boxes, v_labels, v_scores = list(), list(), list()
        for box in boxes:
            for i in range(len(self.__labels)):
                if box.classes[i] > self.__threshold:
                    v_boxes.append(box)
                    v_labels.append(self.__labels[i])
                    v_scores.append(box.classes[i]*100)
        return v_boxes, v_labels, v_scores

    def __draw_prediction(self,
                          img: Image,
                          boxes: List[BoundingBox],
                          labels: List[str],
                          scores: List[float],
                          depth: rs.depth_frame):
        draw = ImageDraw.Draw(img)

        for i in range(len(boxes)):
            box = boxes[i]
            distance = self.__calculate_distance(depth, box)
            draw.rectangle(((box.xmin, box.ymin), (box.xmax, box.ymax)), outline="red")
            label = "%s (%.3f), distance: (%.2f)m" % (labels[i], scores[i], distance)
            draw.text((box.xmin, box.ymin), label)

        return img

    @staticmethod
    def __calculate_distance(depth: rs.depth_frame, box: BoundingBox):
        width = box.xmax - box.xmin
        height = box.ymax - box.ymin

        depth_width = depth.get_width()
        depth_height = depth.get_height()

        middle = (box.xmin + width/2, box.ymin + height/2)

        if depth_width < middle[0] or depth_height < middle[1]:
            print('Bounding box size out of range for depth frame')
            return 0.0

        return depth.get_distance(int(middle[0]), int(middle[1]))

    @staticmethod
    def __preprocess_image(img: Image):
        width, height = img.size
        img = img.resize((416, 416), Image.NEAREST)
        img = img_to_array(img).astype('float32')
        img /= 255.0
        img = expand_dims(img, 0)
        return img, width, height

