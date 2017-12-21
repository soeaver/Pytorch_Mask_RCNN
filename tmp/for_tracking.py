import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from config import Config
from network.mask_rcnn import MaskRCNN
from postprocess import visualize
from tasks.merge_task import final_detections, unmold_detections
from preprocess.InputProcess import (compose_image_meta, mold_image,
                                     mold_inputs, parse_image_meta,
                                     resize_image)



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


class InferenceConfig(Config):

    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable


def run_demo(image_path, visualize_path, txt_save_path, mask_save_path, mask_save_path2, model):
    start = time.time()

    text_file = open(txt_save_path, "w")

    oriImg = cv2.imread(image_path)

    mask_file = np.zeros((oriImg.shape[0], oriImg.shape[1]))

    mask_file2 = np.zeros((oriImg.shape[0], oriImg.shape[1])) 
    image = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

    molded_image, image_metas, windows = mold_inputs([image], config)

    inputs = molded_image.transpose((0, 3, 1, 2))
    inputs = torch.from_numpy(inputs).float()
    inputs = Variable(inputs, volatile=True).cuda()

    outputs = model(inputs)

    rpn_class_logits, rpn_class, rpn_bbox,\
        rpn_rois, mrcnn_class_logits, mrcnn_class,\
        mrcnn_bbox, mrcnn_masks_logits = outputs

    mrcnn_class = mrcnn_class.cpu().data.numpy()
    mrcnn_bbox = mrcnn_bbox.cpu().data.numpy()

    rois = rpn_rois.cpu().data.numpy() / 1024.
    rois = rois[:, :, [1, 0, 3, 2]]

    detections = final_detections(
        rois, mrcnn_class, mrcnn_bbox, image_metas, config)

    mask_rois = detections[..., :4][..., [1, 0, 3, 2]]
    mask_rois = to_variable(mask_rois, volatile=True).cuda()

    mrcnn_mask = model.rpn_mask(model.mrcnn_feature_maps, mask_rois)

    mrcnn_mask = F.sigmoid(mrcnn_mask)
    mrcnn_mask = mrcnn_mask.cpu().data.numpy()
    mrcnn_mask = mrcnn_mask.transpose(0, 1, 3, 4, 2)

    final_rois, final_class_ids, final_scores, final_masks =\
        unmold_detections(detections[0], mrcnn_mask[0],
                          oriImg.shape, windows[0])

    result = {
        "rois": final_rois,
        "class_ids": final_class_ids,
        "scores": final_scores,
        "masks": final_masks,
    }

    person_index = result['class_ids'] == 1
    number_person = np.sum(person_index)

    text_file.write("%d\n" % int(number_person))
    print('number of persons', number_person)

    if number_person != 0:

        person_mask = result['masks'][:, :, person_index]
        person_bbox = result['rois'][person_index]
        person_score = result['scores'][person_index]

        for k in range(number_person):
            score = person_score[k]
            y1, x1, y2, x2 = person_bbox[k]
            w = x2 - x1
            h = y2 - y1

            one_mask = person_mask[:, :, k]
            new_mask = np.ones_like(mask_file) * (k + 1)
            new_mask = new_mask[one_mask.astype(np.bool)]
            mask_file[one_mask.astype(np.bool)] = new_mask

            mask_file2[one_mask.astype(np.bool)] =255
            text_file.write(str(score) + ',' + str(w) + ',' +
                            str(h) + ',' + str(x1) + ',' + str(y1) + '\n')
            print('masks', person_mask.shape)
            print('max number of mask', np.max(person_mask))
            print('scores', person_score)

    text_file.close()

    cv2.imwrite(mask_save_path, mask_file)
    cv2.imwrite(mask_save_path2, mask_file2)
    visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'],
                                class_names, visualize_path, result['scores'])
    end = time.time()
    print('spend time', end - start)


def directory_demo(image_source_path, image_save_path, model):

    images = os.listdir(image_source_path)

    for i, image_path in enumerate(images):
        if i % 10 == 0:
            print('Processed %d images' % i)

        one_source_path = os.path.join(image_source_path, image_path)
        txt_save_path = os.path.join(
            image_save_path, 'bbox', image_path.rsplit('.')[0] + '.txt')
        mask_save_path = os.path.join(
            image_save_path, 'instance_mask', image_path.rsplit('.')[0] + '.png')
        mask_save_path2 = os.path.join(
            image_save_path, 'binary_mask', image_path.rsplit('.')[0] + '.png')            
#        print(mask_save_path)
        visualize_path = os.path.join(
            image_save_path, 'visualize', image_path.rsplit('.')[0] + '.png')
        run_demo(one_source_path, visualize_path,
                 txt_save_path, mask_save_path, mask_save_path2, model)


if __name__ == "__main__":
    config = InferenceConfig()
    config.display()

    pretrained_weight = "/home/tensorboy/AI/walmart/models/mrcnn.pth"
    state_dict = torch.load(pretrained_weight)

    model = MaskRCNN(config=config, mode='inference')
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    cudnn.benchmark = True

    # directory demo, inference on images of one directory and save the result
    image_source_dir = '/home/tensorboy/AI/walmart/DATA/data1'
    tracking_save_dir = '/home/tensorboy/AI/walmart/DATA/data1_result'
    directory_demo(image_source_dir, tracking_save_dir, model)
