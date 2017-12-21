import os
import time

import cv2
import random
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


def run_demo(oriImg,  model):
    start = time.time()
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
    
    return result


if __name__ == "__main__":
    from postprocess import visualize_cap as vc
    config = InferenceConfig()
    config.display()

    pretrained_weight = "/home/tensorboy/AI/walmart/models/mrcnn.pth"
    state_dict = torch.load(pretrained_weight)

    model = MaskRCNN(config=config, mode='inference')
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    cudnn.benchmark = True


    cap = cv2.VideoCapture(1)
    imgindex = 0

    while(True):
        ret, image = cap.read()
        print(image.shape)
        #image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
        # Run detection
        r = run_demo(image, model)

        boxes = r['rois']
        masks = r['masks'] 
        class_ids = r['class_ids'] 
        class_names = class_names
        scores=r['scores']
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        height, width = image.shape[:2]

        # Generate random colors
        colors = vc.random_colors(N)

        masked_image = image.astype(np.uint8).copy()
        for i in range(N):
            color = colors[i]
            class_id = class_ids[i]
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            cv2.imwrite('Mask-Rcnn.jpg',masked_image)
            cv2.rectangle(masked_image, (x1,y1),(x2,y2), (0,255,0),2)
            # Label
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
            # Mask
            mask = masks[:, :, i]
            masked_image = vc.apply_mask(masked_image, mask, color)

            cv2.putText(masked_image, caption, (int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,255),2)
        print('mask shape', masked_image.shape)
        cv2.imshow('Mask-Rcnn',masked_image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            imgindex += 1
            cv2.imwrite('./{0}.jpg'.format(imgindex),masked_image)
    cap.release()
    cv2.destroyAllWindows()

