# -*- coding: utf-8 -*-
# @Time    : 2018/3/6 0:49
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : predict_video.py
# @Software: PyCharm

import os
import imageio
from keras import backend as K
from keras.models import load_model
from utils.utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.models.keras_yolo import yolo_head
from tqdm import tqdm
from models.YOLO_FUNCTION import yolo_eval


sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)

yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


def predict_video(sess, video_file, video_out_file):
    video_in = imageio.get_reader(video_file)
    frames = []
    for i, image in enumerate(tqdm(video_in)):
        imageio.imwrite("data/cache.jpg",image)
        image_in, image_data = preprocess_image("data/cache.jpg", model_image_size=(608, 608),image_shape=(720,1280),type="video")
        out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                      feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
        colors = generate_colors(class_names)
        draw_boxes(image_in, out_scores, out_boxes, out_classes, class_names, colors)
        image_in.save(os.path.join("data", "cache_out.jpg"), quality=90)
        frames.append(imageio.imread("data/cache_out.jpg"))
    imageio.mimsave(video_out_file, frames)

predict_video(sess, "data/input/video/test_video2.mp4", 'data/output/video/movie.avi')