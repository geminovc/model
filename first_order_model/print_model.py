import torch

import sys
from shrink_util import print_gen_module, print_diff

"""
Prints out the shapes of a model if run as python print_model.py model_state_dict

If run as python print_model.py state_dict1 state_dict2. print out the diff between model shapes
"""
if __name__ == '__main__':
    if len(sys.argv) == 2:
        state_dict = torch.load(sys.argv[1])
        print_gen_module(state_dict)
    else:
        print("FIRST SHAPE | SECOND SHAPE | NAME")
        state_dict = torch.load(sys.argv[1])
        state_dict2 = torch.load(sys.argv[2])
        print_diff(state_dict, state_dict2)

