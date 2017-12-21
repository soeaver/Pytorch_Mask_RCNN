import math
import os
import random
import sys
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.io

import coco
import model as modellib
import utils
import visualize
from preprocess.ImageProcess import (mold_inputs, resize_image)
# Root directory of the project
import tensorflow as tf
# Directory to save logs and trained model
# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = "/home/tensorboy/AI/walmart/models/MASK_RCNN/coco20171107T0332/mask_rcnn_coco_0529.h5"

# Directory of images to run detection on

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


image_path = 'README/newyork.jpg'
save_path = 'README/newyork_output.jpg'
oriImg = cv2.imread(image_path)
image = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
print(image.shape)
# Run detection
results = model.detect([image], verbose=1)
# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, save_path, r['scores'])
                            
                         
roi_pillar = model.keras_model.get_layer("roi_align_classifier").output
gather_nd0 = model.ancestor(roi_pillar, "roi_align_classifier/gather_nd0")

pillar = model.keras_model.get_layer("ROI").output
nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")

if nms_node is None:
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")

roi=[n.name for n in tf.get_default_graph().as_graph_def().node]

roi_dicts = []
for name in roi:
    if 'roi' in name:
        new_name = name.replace('/','_')
        roi_dicts.append((new_name, model.keras_model.get_layer(name).output))


mrcnn = model.run_graph([image], [

    ("input_image",        model.keras_model.get_layer("input_image").output),
    ("bn_conv1",              model.keras_model.get_layer("bn_conv1").output), 
    ("res2c_out",          model.keras_model.get_layer("res2c_out").output), 
    ("res3d_out",          model.keras_model.get_layer("res3d_out").output),         
    ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  
    ('res5c_out',          model.keras_model.get_layer("res5c_out").output),
    ("fpn_c5p5",           model.keras_model.get_layer("fpn_c5p5").output), 
    ("fpn_p4add",           model.keras_model.get_layer("fpn_p4add").output), 
    ("fpn_p3add",           model.keras_model.get_layer("fpn_p3add").output), 
    ("fpn_p2add",           model.keras_model.get_layer("fpn_p2add").output),
    ("fpn_p6",           model.keras_model.get_layer("fpn_p6").output),    
    ("fpn_p5",           model.keras_model.get_layer("fpn_p5").output),    
    ("fpn_p4",           model.keras_model.get_layer("fpn_p4").output),
    ("fpn_p3",           model.keras_model.get_layer("fpn_p3").output),     
    ("fpn_p2",           model.keras_model.get_layer("fpn_p2").output),
    ("roi",                model.keras_model.get_layer("ROI").output),
    ("roi_align_classifier", model.keras_model.get_layer("roi_align_classifier").output),    
    ("roi_align_mask",  model.keras_model.get_layer("roi_align_mask").output), 
    ("rpn_class", model.keras_model.get_layer("rpn_class").output),
    ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),    
    ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
    ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
    ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
    ("post_nms_anchor_ix", nms_node),
    ("proposals", model.keras_model.get_layer("ROI").output),
    ("mrcnn_class_conv1",model.keras_model.get_layer("mrcnn_class_conv1").output),  
    ("mrcnn_class_bn1", model.keras_model.get_layer("mrcnn_class_bn1").output), 
    ("mrcnn_class_conv2",model.keras_model.get_layer("mrcnn_class_conv2").output),    
    ("mrcnn_class_bn2", model.keras_model.get_layer("mrcnn_class_bn2").output),   
    ("pool_squeeze", model.keras_model.get_layer("pool_squeeze").output),    
    ("mrcnn_class_logits", model.keras_model.get_layer("mrcnn_class_logits").output), 
    ("mrcnn_class", model.keras_model.get_layer("mrcnn_class").output),
    ("mrcnn_bbox", model.keras_model.get_layer("mrcnn_bbox").output),
    ("mrcnn_detection", model.keras_model.get_layer("mrcnn_detection").output),   
    ("mrcnn_mask", model.keras_model.get_layer("mrcnn_mask").output),    
    ("roi_align_mask/gather_nd1",model.keras_model.get_layer("roi_align_mask/gather_nd1").output),      
])

result_save_dir = '/home/tensorboy/AI/walmart/intermediate_result/keras'

for k,v in mrcnn.items():
    save_path = os.path.join(result_save_dir, k)+'.npy'
    np.save(save_path, v)
