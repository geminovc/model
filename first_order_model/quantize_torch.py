import sys
sys.path.append("..")
from keypoint_based_face_models import KeypointBasedFaceModels
from fom_wrapper import FirstOrderModel
import numpy as np
import time
import torch

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
    model_fp32 = model.generator
    start_time = time.time()
    res = model_fp32(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})    
    print("Inference on float32:", time.time()-start_time)
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # model_fp32_fused = torch.quantization.fuse_modules(model_fp32, ['conv'])
    model_fp32_prepared = model_fp32 #torch.quantization.prepare(model_fp32_fused)
    model_fp32_prepared(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})
    # model_int8 = torch.quantization.convert(model_fp32_prepared)
    model_int8 = torch.quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)

# run the model
for i in range(0, 100):
    start_time = time.time()
    res = model_int8(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})
    print("Inference on int8:", time.time()-start_time)