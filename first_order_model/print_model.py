import torch

import sys
from shrink_util import print_gen_module, print_diff

"""
Prints out the shapes of a model if run as python print_model.py model_state_dict

If run as python print_model.py state_dict1 state_dict2. print out the diff between model shapes

Model state dict is the state dictionary of the model. See: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html#save-and-load-the-model-via-state-dict

In general the work flow is
1) Generate state dict for two models using method described in link.
2) Feed those into this file
"""
if __name__ == '__main__':
    if len(sys.argv) == 2:

        # In the one argument case, just prints the generator state dict
        state_dict = torch.load(sys.argv[1])
        print_gen_module(state_dict)
    else:
        print("FIRST SHAPE | SECOND SHAPE | NAME")

        # Load both state dicts
        state_dict = torch.load(sys.argv[1])
        state_dict2 = torch.load(sys.argv[2])

        # Prints a diff of the two state dicts
        print_diff(state_dict, state_dict2)
