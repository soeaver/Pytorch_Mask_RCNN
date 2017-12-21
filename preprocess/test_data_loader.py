import numpy as np
import torch
from torch.autograd import Variable
from preprocess.data_center import CocoDataset
from preprocess.data_generator import data_generator
from config import Config

def to_variable(numpy_data, cuda=True):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data)
    
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
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    
class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

dataset_path = '/home/tensorboy/AI/walmart/DATA'
#dataset_train = CocoDataset()
#dataset_train.load_coco(args.dataset, "train")
#dataset_train.load_coco(args.dataset, "val35k")
#dataset_train.prepare()

# Validation dataset
dataset_val = CocoDataset()
dataset_val.load_coco(dataset_path, "minival")
dataset_val.prepare()

# Data generators
#train_generator = data_generator(train_dataset, config, shuffle=True, 
#                                        batch_size=config.BATCH_SIZE)
                                        
val_generator = data_generator(dataset_val, config, shuffle=False, 
                                    batch_size=config.BATCH_SIZE)

while True:
    inputs, labels = next(val_generator)
    batch_images, batch_image_meta, \
    batch_rpn_match, batch_rpn_bbox,\
    batch_gt_boxes, batch_gt_masks = inputs
    
    batch_images = batch_images.transpose(0,3,1,2)
    
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
    


