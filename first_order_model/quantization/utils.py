import torch
import time, os, sys
import csv

QUANT_ENGINE = 'fbgemm'

def get_variance(test_list, mean):
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance ** 0.5
    return res


def print_average_and_std(test_list, name):
    mean = sum(test_list) / len(test_list)
    res = get_variance(test_list, mean)
    print(f"{name}:: mean={round(mean, 6)}, std={round(res / mean * 100, 6)}%")
    return mean, res


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


def print_model_info(model, model_name, x0, x1=None, x2=None, x3=None, x4=None, num_runs=100):
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
    return ['conv', 'norm', 'relu']


def get_coder_modules_to_fuse(num_blocks, prefix='down_blocks'):
    modules = []
    basics = get_basic_module_to_fuse()
    for i in range(0, num_blocks):
        new_module = []
        for sub_module in basics:
            new_module.append(prefix + '.' + str(i) + '.' + sub_module)
        modules.append(new_module)
    
    return modules

def display_times(times, module_name, USE_FAST_CONV2, USE_QUANTIZATION, USE_FLOAT_16, IMAGE_RESOLUTION):
    print(f"using custom conv:{USE_FAST_CONV2}, using quantization:{USE_QUANTIZATION}")
    print(f"using float16:{USE_FLOAT_16}, resolution:{IMAGE_RESOLUTION}")
    for key in times.keys():
        print_average_and_std(times[key], key)

    if USE_FAST_CONV2:
        module_name += '_fast_conv'
    if USE_QUANTIZATION:
        module_name += '_int8'
    if USE_FLOAT_16:
        module_name += '_float16'
    if IMAGE_RESOLUTION:
        module_name += f'_res{IMAGE_RESOLUTION}'

    with open(module_name + '.csv', 'w', encoding='UTF8') as f:
        header = ['measurement']
        header += [key for key in times.keys()]
        writer = csv.writer(f)
        writer.writerow(header)
        mean_row = ['mean']
        mean_row += [sum(times[key])/len(times[key]) for key in times.keys()]
        std_row = ['std%']
        std_row += [100 * get_variance(times[key], sum(times[key])/len(times[key])) / (sum(times[key])/len(times[key])) for key in times.keys()]
        writer.writerow(mean_row)
        writer.writerow(std_row)



 