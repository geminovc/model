import torch
import time, os, sys


QUANT_ENGINE = 'fbgemm'


def print_average_and_std(test_list, name):
    mean = sum(test_list) / len(test_list)
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance ** 0.5
    print(f"{name}:: mean={round(mean, 6)}, std={round(res / mean * 100, 6)}%")


def get_params(model):
    params = []
    for name, mod in model.named_modules():                             
        if isinstance(mod, torch.nn.quantized.Conv2d):                              
            weight, bias = mod._weight_bias()
            params.append(weight)
            params.append(bias)
    return params


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size


def print_model_info(model, model_name, x0, x1=None, x2=None, x3=None, x4=None, num_runs=10):
        print_size_of_model(model, label=model_name)
        tt = []
        for i in range(0, num_runs):
            print(f"run #{i}")
            if x1 != None:
                start_time = time.time()
                res = model(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})
                tt.append(time.time() - start_time)
            else:
                start_time = time.time()
                res = model(x0)
                tt.append(time.time() - start_time)
        print_average_and_std(tt, f"Average inference time on {model_name}")


def quantize_model(model_fp32, modules_to_fuse, x0, x1=None, x2=None, x3=None, x4=None, enable_meausre=False):
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig(QUANT_ENGINE)
    model_fp32_fused = torch.quantization.fuse_modules(model_fp32, modules_to_fuse)
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
    model_int8 = torch.quantization.convert(model_fp32_prepared)

    if enable_meausre:
        print_model_info(model_fp32, "model_fp32", x0, x1, x2, x3, x4)
        print_model_info(model_int8, "model_int8", x0, x1, x2, x3, x4)

    return model_int8


def get_basic_module_to_fuse():
    return [['conv', 'norm', 'relu']]
 