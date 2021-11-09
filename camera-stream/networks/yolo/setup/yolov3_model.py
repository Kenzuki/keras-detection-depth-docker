from yolov3_util import make_yolov3_model, WeightReader

model = make_yolov3_model()

weight_reader = WeightReader('../yolov3.weights')

weight_reader.load_weights(model)

model.save('../model.h5')

