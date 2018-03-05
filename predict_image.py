# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 9:43
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : predict.py
# @Software: PyCharm


# coding: utf-8

# # Autonomous driving - Car detection

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pylab
import scipy.io
import scipy.misc

from keras import backend as K
from keras.models import load_model
from utils.utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.models.keras_yolo import yolo_head
from models.YOLO_FUNCTION import yolo_eval

sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)

yolo_model = load_model("model_data/yolo.h5")
# yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def predict_image(sess, image_in_file, image_out_file):
    image, image_data = preprocess_image(image_in_file, model_image_size=(608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    print('Found {} boxes for {}'.format(len(out_boxes), image_in_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(image_out_file, quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(image_out_file)
    imshow(output_image)
    pylab.show()
    return out_scores, out_boxes, out_classes

out_scores, out_boxes, out_classes = predict_image(sess, "data/input/image/test.jpg", 'data/output/image/test_out.jpg')




