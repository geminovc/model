import torch

import sys
from shrink_util import print_gen_module, print_diff

# Expects the module path in second argument
if __name__ == '__main__':
    if len(sys.argv) == 2:
        state_dict =torch.load(sys.argv[1])
        print_gen_module(state_dict)
    else:
        print("FIRST SHAPE | SECOND SHAPE | NAME")
        state_dict =torch.load(sys.argv[1])
        state_dict2 =torch.load(sys.argv[2])
        print_diff(state_dict, state_dict2)

