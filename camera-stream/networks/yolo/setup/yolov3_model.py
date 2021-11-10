from ..yolov3_util import make_yolov3_model, WeightReader

model = make_yolov3_model()

weight_reader = WeightReader('camera-stream/networks/yolo/setup/yolov3.weights')

weight_reader.load_weights(model)

model.save('camera-stream/networks/yolo/model.h5')

