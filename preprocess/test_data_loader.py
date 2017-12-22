import numpy as np
import torch
from torch.autograd import Variable
from preprocess.data_center import CocoDataset
from preprocess.coco_data_pipeline import CocoLoader
from config import Config

def to_variable(torch_data, cuda=True):
    
    variable = Variable(torch_data)
    
    if cuda:
        variable.cuda()
    return variable

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    
config = CocoConfig()
config.display()

dataset_path = '/home/tensorboy/AI/walmart/DATA'
dataset_train = CocoDataset()
dataset_train.load_coco(dataset_path, "train")
dataset_train.load_coco(dataset_path, "val35k")
dataset_train.prepare()

# Validation dataset
dataset_val = CocoDataset()
dataset_val.load_coco(dataset_path, "minival")
dataset_val.prepare()


train_data = CocoLoader(dataset_train, config, shuffle=True, augment=True, batch_size = 1, num_workers = 6)
print('train dataset len: {}'.format(len(train_data.dataset)))

# validation data
valid_data = CocoLoader(dataset_val, config, shuffle=False, augment=False, batch_size = 1, num_workers = 4)
print('val dataset len: {}'.format(len(valid_data.dataset)))


while True:
    batch_images, batch_image_meta, \
    batch_rpn_match, batch_rpn_bbox,\
    batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = next(valid_data.get_stream())
    batch_images = np.transpose(batch_images, (0,3,1,2))
    

    images = to_variable(batch_images)
    metas = to_variable(batch_image_meta)
    rpns =  to_variable(batch_rpn_match)
    rpn_boxes = to_variable(batch_rpn_bbox)
    gt_boxes = to_variable(batch_gt_boxes)    
    gt_masks = to_variable(batch_gt_masks)

    print images.size()
    print metas.size()
    print rpns.size()
    print rpn_boxes.size()
    print gt_boxes.size()
    print gt_masks.size()
    


