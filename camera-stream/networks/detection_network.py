from PIL import Image, ImageDraw
from abc import ABC, abstractmethod
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model
from numpy import expand_dims
from typing import List
from networks.util import calculate_distance
import pyrealsense2 as rs


class DetectionNetwork(ABC):

    def __init__(self, lang: str):
        self._model = self._make_model()
        self._labels = self._load_labels(lang)

    def predict(self, rgb_frame: Image, depth_frame: rs.depth_frame) -> Image:
        img, img_w, img_h = DetectionNetwork.__preprocess_image(rgb_frame)
        boxes, scores, classes, nums = self._model.predict(img)
        return DetectionNetwork.__draw_prediction(rgb_frame, depth_frame, (boxes, scores, classes, nums), self._labels)

    @staticmethod
    @abstractmethod
    def _make_model() -> Model:
        pass

    @staticmethod
    @abstractmethod
    def _load_labels(lang: str) -> List[str]:
        pass

    @staticmethod
    def __preprocess_image(img: Image):
        width, height = img.size
        img = img.resize((416, 416), Image.NEAREST)
        img = img_to_array(img).astype('float32')
        img /= 255.0
        img = expand_dims(img, 0)
        return img, width, height

    @staticmethod
    def __draw_prediction(img: Image, depth: rs.depth_frame, outputs, labels: List[str]):
        boxes, scores, classes, nums = outputs
        boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]
        wh = img.size
        draw = ImageDraw.Draw(img)

        for i in range(nums):
            box = boxes[i]
            x1y1 = tuple((box[0:2]) * wh)
            x2y2 = tuple((box[2:4]) * wh)
            distance = calculate_distance(depth, x1y1, x2y2)
            draw.rectangle((x1y1, x2y2), outline="red")
            label = "%s (%.3f), distance: (%.2f)m" % (labels[int(classes[i])], scores[i], distance)
            draw.text((x1y1[0], x1y1[1]), label)

        return img

