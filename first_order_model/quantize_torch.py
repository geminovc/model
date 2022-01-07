from first_order_model.fom_wrapper import FirstOrderModel
import numpy as np
import time
import torch
import os

# get model size
def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

model = FirstOrderModel("config/api_sample.yaml")

x0 = torch.randn(1, 3, 256, 256, requires_grad=False)
x1 = torch.randn(1, 10, 2,requires_grad=False)
x2 = torch.randn(1, 10, 2, 2, requires_grad=False)
x3 = torch.randn(1, 10, 2, requires_grad=False)
x4 = torch.randn(1, 10, 2, 2, requires_grad=False)

convert_kp_extractor = False
convert_generator = True
dynamic = False

if convert_generator:
    model_fp32 = model.generator
    start_time = time.time()
    res = model_fp32(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})    
    print("Inference on float32:", time.time() - start_time)
    model_fp32.eval()
    if dynamic:
        model_int8 = torch.quantization.quantize_dynamic(model_fp32, dtype=torch.qint8)
    else:
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.backends.quantized.engine = 'fbgemm'
        # model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [])
        model_fp32_prepared = model_fp32 #torch.quantization.prepare(model_fp32_fused)
        model_int8 = torch.quantization.convert(model_fp32_prepared)

# compare the sizes
f = print_size_of_model(model_fp32,"fp32")
q = print_size_of_model(model_int8,"int8")
print("int8 is {0:.2f} times smaller than fp32".format(f/q))

# run the model
for i in range(0, 100):
    start_time = time.time()
    res = model_int8(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})
    print("Inference on int8:", time.time() - start_time)
