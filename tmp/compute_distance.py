import os
import numpy as np

output_names = ["roi_align_classifier", "roi_align_mask", "fpn_c5p5","fpn_p4add","fpn_p3add","fpn_p2add", "fpn_p6", "fpn_p5", "fpn_p4","fpn_p3", "fpn_p2",  "roi", "rpn_class", "rpn_class_logits","rpn_bbox", "pre_nms_anchors",  "refined_anchors", "refined_anchors_clipped", "post_nms_anchor_ix", "proposals", "mrcnn_class", "mrcnn_bbox", "mrcnn_class_bn2", "mrcnn_class_bn1", "mrcnn_class_conv1","mrcnn_class_conv2", "mrcnn_class_logits", "pool_squeeze", "mrcnn_mask", "mrcnn_detection"]   
    
     
result_save_dir = '/home/tensorboy/AI/walmart/intermediate_result'
    
keras_output   = np.load(os.path.join(result_save_dir, 'keras', output_names[12]+'.npy'))
pytorch_output = np.load(os.path.join(result_save_dir, 'pytorch', output_names[12]+'.npy'))

print(keras_output.shape)
print(pytorch_output.shape)

print('output have %10.10f%% relative error'%(np.linalg.norm(keras_output-pytorch_output)/np.linalg.norm(keras_output)*100))
