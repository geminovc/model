import sys
sys.path.append("..")
from keypoint_based_face_models import KeypointBasedFaceModels
from fom_wrapper import FirstOrderModel
import numpy as np
import time
import torch
import os
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

model = FirstOrderModel("config/api_sample.yaml")

x0 = torch.randn(1, 3, 256, 256, requires_grad=False)
x1 = torch.randn(1, 10, 2,requires_grad=False)
x2 = torch.randn(1, 10, 2, 2, requires_grad=False)
x3 = torch.randn(1, 10, 2, requires_grad=False)
x4 = torch.randn(1, 10, 2, 2, requires_grad=False)
# x_out = torch.randn(1, 3, 256, 256, requires_grad=False)

convert_kp_extractor = False
convert_generator = True

if convert_generator:
    module = model.generator
    module.eval()
    start_time = time.time()
    res = module(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})    
    print("Inference on float32:", time.time()-start_time)
    
    for i in range(0, len(module.dense_motion_network.hourglass.encoder.down_blocks)):
        module.dense_motion_network.hourglass.encoder.down_blocks[i].norm = \
        prune.random_unstructured(module.dense_motion_network.hourglass.encoder.down_blocks[i].norm, name="weight", amount=0)
        module.dense_motion_network.hourglass.encoder.down_blocks[i].norm = \
        prune.random_unstructured(module.dense_motion_network.hourglass.encoder.down_blocks[i].norm, name="bias", amount=0)

        module.dense_motion_network.hourglass.decoder.up_blocks[i].norm = \
        prune.random_unstructured(module.dense_motion_network.hourglass.decoder.up_blocks[i].norm, name="weight", amount=0)
        module.dense_motion_network.hourglass.decoder.up_blocks[i].norm = \
        prune.random_unstructured(module.dense_motion_network.hourglass.decoder.up_blocks[i].norm, name="bias", amount=0)

# run the model
for i in range(0, 100):
    start_time = time.time()
    res = module(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})
    print("Inference on pruned model:", time.time()-start_time)
