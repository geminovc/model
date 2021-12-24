from first_order_model.fom_wrapper import FirstOrderModel
import numpy as np
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = FirstOrderModel("../config/api_sample.yaml", True)

x0 = torch.randn(1, 3, 256, 256, requires_grad=False)
x1 = torch.randn(1, 10, 2, requires_grad=False)
x2 = torch.randn(1, 10, 2, 2, requires_grad=False)
x3 = torch.randn(1, 10, 2, requires_grad=False)
x4 = torch.randn(1, 10, 2, 2, requires_grad=False)

convert_kp_extractor = True
convert_generator = True
if convert_generator:

    torch.onnx.export(model.generator,           # model being run 
                      (x0, x1, x2, x3, x4),      # model input (or a tuple for multiple inputs)
                      "fom_gen.onnx",            # where to save the model 
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['source_image', 'kp_driving_v', 'kp_driving_j',
                                     'kp_source_v', 'kp_source_j'],
                      output_names = ['output']) # the model's output names
if convert_kp_extractor:
    torch.onnx.export(model.kp_detector,         # model being run
                      x0,                        # model input (or a tuple for multiple inputs)
                      "fom_kp.onnx",             # where to save the model 
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['source'],  # the model's input names
                      output_names = ['output']) # the model's output names