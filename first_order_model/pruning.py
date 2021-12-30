import sys
sys.path.append("..")
from keypoint_based_face_models import KeypointBasedFaceModels
from fom_wrapper import FirstOrderModel
import numpy as np
import time
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy


def prune_model_l1_unstructured(model, layer_type, proportion):
    for module in model.modules():
        try:
            if isinstance(module, layer_type):
                prune.random_unstructured(module, 'weight', proportion)
                prune.remove(module, 'weight')
        except:
            pass
    return model


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):
    num_zeros = 0
    num_elements = 0
    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer < 0.1).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer < 0.1).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param < 0.01).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param < 0.01).item()
                num_elements += param.nelement()

    if num_elements != 0:
        sparsity = num_zeros / num_elements
    else:
        sparsity = 0

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model, weight=True, bias=False, conv2d_use_mask=False, linear_use_mask=False):
    num_zeros = 0
    num_elements = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements
        elif isinstance(module, torch.nn.Linear):
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    if num_elements != 0:
        sparsity = num_zeros / num_elements
    else:
        sparsity = 0

    return num_zeros, num_elements, sparsity


def model_pruning(model, conv2d_prune_amount=0.4, linear_prune_amount=0.2, grouped_pruning=True):
    num_zeros, num_elements, sparsity = measure_global_sparsity(model, weight=True, bias=True, conv2d_use_mask=False, linear_use_mask=False)

    print("Global Sparsity before pruning:")
    print("{:.2f}".format(sparsity))

    if grouped_pruning == True:
        # Global pruning
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, "weight"))
                parameters_to_prune.append((module, "bias"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=conv2d_prune_amount,
        )
    else:
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module,
                                        name="weight",
                                        amount=conv2d_prune_amount)
                prune.l1_unstructured(module,
                                        name="bias",
                                        amount=conv2d_prune_amount)
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module,
                                        name="weight",
                                        amount=linear_prune_amount)
                prune.l1_unstructured(module,
                                        name="bias",
                                        amount=linear_prune_amount)                

    num_zeros, num_elements, sparsity = measure_global_sparsity(model, weight=True, bias=False, conv2d_use_mask=False, linear_use_mask=False)

    print("Global Sparsity after pruning:")
    print("{:.2f}".format(sparsity))

    return model


def time_the_model(model):
    x0 = torch.randn(1, 3, 256, 256, requires_grad=False)
    x1 = torch.randn(1, 10, 2,requires_grad=False)
    x2 = torch.randn(1, 10, 2, 2, requires_grad=False)
    x3 = torch.randn(1, 10, 2, requires_grad=False)
    x4 = torch.randn(1, 10, 2, 2, requires_grad=False)
    for i in range(0, 10):
        start_time = time.time()
        res = model(x0, {'value':x1, 'jacobian':x2}, {'value':x3, 'jacobian':x4})
        print("Inference on model:", time.time() - start_time)

def remove_parameters(model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model

def main():
    model_fom = FirstOrderModel("config/api_sample.yaml")
    convert_kp_extractor = False
    convert_generator = True

    if convert_generator:
        model = model_fom.generator
        model.eval()
        # print("Timing the original:")
        # time_the_model(model)

        pruned_model = copy.deepcopy(model)
        pruned_model.eval()
        pruned_model = model_pruning(model=pruned_model, conv2d_prune_amount=0.001, linear_prune_amount=0.002, grouped_pruning=False)
        pruned_model = remove_parameters(model=pruned_model)
        print("Timing the pruned:")
        time_the_model(pruned_model)


if __name__ == "__main__":
    main()
