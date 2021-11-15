import tensorflow as tf


def meshgrid(n_a, n_b):
    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]


def yolo_boxes(pred, anchors, classes):
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    grid = meshgrid(grid_size[1], grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, classes):
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    if classes == 1:
        scores = confidence
    else:
        scores = confidence * class_probs

    dscores = tf.squeeze(scores, axis=0)
    scores = tf.reduce_max(dscores, [1])
    bbox = tf.reshape(bbox, (-1, 4))
    classes = tf.argmax(dscores, 1)
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=10,
        iou_threshold=0.5,
        score_threshold=0.5,
        soft_nms_sigma=0.5)
    num_valid_nms_boxes = tf.shape(selected_indices)[0]

    selected_indices = tf.concat([selected_indices, tf.zeros(10 - num_valid_nms_boxes, tf.int32)], 0)
    selected_scores = tf.concat([selected_scores, tf.zeros(10 - num_valid_nms_boxes, tf.float32)], -1)

    boxes = tf.gather(bbox, selected_indices)
    boxes = tf.expand_dims(boxes, axis=0)
    scores = selected_scores
    scores = tf.expand_dims(scores, axis=0)
    classes = tf.gather(classes, selected_indices)
    classes = tf.expand_dims(classes, axis=0)
    valid_detections = num_valid_nms_boxes
    valid_detections = tf.expand_dims(valid_detections, axis=0)

    return boxes, scores, classes, valid_detections
