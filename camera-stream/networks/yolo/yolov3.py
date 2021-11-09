from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw
from numpy import expand_dims
from networks.yolo.yolov3_util import decode_netout, correct_yolo_boxes, do_nms
import io


class YoloNetwork:
    __anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    __threshold = 0.6
    __labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def __init__(self) -> None:
        self.model = load_model('./networks/yolo/model.h5', compile=False)

    @staticmethod
    def __preprocess_image(img_bytes: bytes):
        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size
        img = img.convert('RGB')
        img = img.resize((416, 416), Image.NEAREST)
        img = img_to_array(img).astype('float32')
        img /= 255.0
        img = expand_dims(img, 0)
        return img, width, height

    def __get_boxes(self, boxes):
        v_boxes, v_labels, v_scores = list(), list(), list()
        for box in boxes:
            for i in range(len(self.__labels)):
                if box.classes[i] > self.__threshold:
                    v_boxes.append(box)
                    v_labels.append(self.__labels[i])
                    v_scores.append(box.classes[i]*100)
        return v_boxes, v_labels, v_scores

    def __draw_prediction(self, img, boxes, labels, scores, depth):
        img_obj = Image.open(io.BytesIO(img))
        img_obj = img_obj.convert('RGB')
        draw = ImageDraw.Draw(img_obj)

        for i in range(len(boxes)):
            box = boxes[i]
            y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
            distance = self.__calculate_distance(depth, y1, x1, y2, x2)
            draw.rectangle(((x1, y1), (x2, y2)), outline="red")
            label = "%s (%.3f), distance: (%.2f)m" % (labels[i], scores[i], distance)
            draw.text((x1, y1), label)

        buf = io.BytesIO()
        img_obj.save(buf, format="jpeg")
        return buf.getvalue()

    @staticmethod
    def __calculate_distance(depth, y1, x1, y2, x2):
        width = x2 - x1
        height = y2 - y1

        depth_width = depth.get_width()
        depth_height = depth.get_height()

        middle = (x1 + width/2, y1 + height/2)

        if depth_width < middle[0] or depth_height < middle[1]:
            print('Bounding box size out of range for depth frame')
            return 0

        return depth.get_distance(int(middle[0]), int(middle[1]))

    def predict(self, img: bytes, depth):
        image, img_w, img_h = self.__preprocess_image(img)
        yhat = self.model.predict(image)
        boxes = list()
        for i in range(len(yhat)):
            boxes += decode_netout(yhat[i][0], self.__anchors[i], self.__threshold, 416, 416)
        correct_yolo_boxes(boxes, img_h, img_w, 416, 416)
        v_boxes, v_labels, v_scores = self.__get_boxes(boxes)
        selected_boxes, selected_labels, selected_scores = do_nms(v_boxes, v_labels, v_scores, 0.5)
        for i in range(len(selected_boxes)):
            print(selected_labels[i], selected_scores[i])
        return self.__draw_prediction(img, selected_boxes, selected_labels, selected_scores, depth)
