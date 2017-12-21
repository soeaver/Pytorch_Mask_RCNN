import os
import time

import numpy as np
import skimage
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from config import Config
from network.mask_rcnn import MaskRCNN
from tasks.merge_task import final_detections, unmold_detections
from preprocess.data_center import CocoDataset
from preprocess.InputProcess import (compose_image_meta, mold_image,
                                     mold_inputs, parse_image_meta,
                                     resize_image)

from tnn.network.net_utils import load_net

def to_variable(numpy_data, volatile=False, is_cuda=True):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    if is_cuda:
        variable = variable.cuda()
    return variable

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
            
############################################################
#  COCO Evaluation
############################################################
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, config, eval_type="bbox", limit=None, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]
        
    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        if i%10==0:
            print('Processed %d images'%i )
        # Load image
        image = dataset.load_image(image_id)
        # Run detection
        t = time.time()
        r = inference(image, model, config)
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    # Only evaluate for person.
    cocoEval.params.catIds = coco.getCatIds(catNms=['person']) 
    cocoEval.evaluate()
    a=cocoEval.accumulate()
    b=cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
    

def inference(image, model, config):

    molded_image, image_metas, windows = mold_inputs([image], config)

    inputs = np.transpose(molded_image, (0, 3, 1, 2))
    inputs = to_variable(inputs, volatile=True, is_cuda=True)
    outputs = model(inputs)

    rpn_class_logits, rpn_class, rpn_bbox,\
        rpn_rois, mrcnn_class_logits, mrcnn_class,\
        mrcnn_bbox, mrcnn_masks_logits = outputs

    mrcnn_class = mrcnn_class.cpu().data.numpy()
    mrcnn_bbox = mrcnn_bbox.cpu().data.numpy()

    rois = rpn_rois.cpu().data.numpy() / float(config.IMAGE_MAX_DIM)
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
                          image.shape, windows[0])
    
    result = {
        "rois": final_rois,
        "class_ids": final_class_ids,
        "scores": final_scores,
        "masks": final_masks,
    }

    return result


if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Validation Mask R-CNN on MS COCO.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")

    args = parser.parse_args()
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)

    config = InferenceConfig()
    config.display()

    pretrained_weight = "/extra/tensorboy/pretrained_models/mrcnn.pth"
    state_dict = torch.load(pretrained_weight)
#    meta = load_net(pretrained_weight, model)

    model = MaskRCNN(config=config, mode='inference')
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    cudnn.benchmark = True

    # Validation dataset
    dataset_val = CocoDataset()
    coco = dataset_val.load_coco("/data/coco", "minival", return_coco=True)
    dataset_val.prepare()
    #"bbox" or "segm" for bounding box or segmentation evaluation
    evaluate_coco(model, dataset_val, coco, config, "bbox")
