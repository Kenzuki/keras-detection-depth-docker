import logging
import numpy as np

from tensorflow.keras import Model
from networks.yolo.yolo import YoloNetwork

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2'
]


def load_darknet_weights(model: Model, weights_file):
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        for layer_name in YOLOV3_LAYER_LIST:
            sub_model = model.get_layer(layer_name)
            for i, layer in enumerate(sub_model.layers):
                if not layer.name.startswith('conv2d'):
                    continue
                batch_norm = None
                if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
                    batch_norm = sub_model.layers[i + 1]

                logging.info("{}/{} {}".format(sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

                filters = layer.filters
                size = layer.kernel_size[0]
                in_dim = layer.get_input_shape_at(0)[-1]

                if batch_norm is None:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                else:
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

                conv_shape = (filters, in_dim, size, size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if batch_norm is None:
                    layer.set_weights([conv_weights, conv_bias])
                else:
                    layer.set_weights([conv_weights])
                    batch_norm.set_weights(bn_weights)

        assert len(wf.read()) == 0, 'failed to read all data'


if __name__ == "__main__":
    model = YoloNetwork(lang='en').get_model()
    model.summary()
    logging.info('model created')

    load_darknet_weights(model, 'yolov3.weights')
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = model(img)
    logging.info('sanity check passed')

    model.save_weights('yolov3.tf')
    logging.info('weights saved')
