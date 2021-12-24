import numpy as np
import onnxruntime
import torch
import time

x0 = torch.randn(1, 3, 256, 256, requires_grad=False)
x1 = torch.randn(1, 10, 2, requires_grad=False)
x2 = torch.randn(1, 10, 2, 2, requires_grad=False)
x3 = torch.randn(1, 10, 2, requires_grad=False)
x4 = torch.randn(1, 10, 2, 2, requires_grad=False)
ort_session = onnxruntime.InferenceSession("fom_gen.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Generator
# Jacobians are off for now
ort_inputs = {'source_image': to_numpy(x0),
'kp_driving_v': to_numpy(x1),
# 'kp_driving_j': to_numpy(x2),
'kp_source_v': to_numpy(x3)}
# 'kp_source_j': to_numpy(x4)}

ss = time.time()
ort_outs = ort_session.run(None, ort_inputs)
print("Generator took:",time.time() - ss)

# KP detector
ort_session = onnxruntime.InferenceSession("fom_kp.onnx")

ort_inputs = {'source': to_numpy(x0)}
ss = time.time()
ort_outs = ort_session.run(None, ort_inputs)
print("KP detector took:",time.time() - ss)