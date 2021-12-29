from first_order_model.fom_wrapper import FirstOrderModel
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_model_info(model):
    modules = [module for module in model.modules()]
    modules_params = []
    for module in modules:
        children_param = []
        # print("module############################################################################")
        # print(module)
        # print("children-------------------------------------------------------------------------")
        children = get_children(module)
        try:
            for child in children:
                # print(child)
                children_param.append(get_n_params(child))
        except:
            children_param.append(get_n_params(children))
        modules_params.append(children_param)
    return modules_params

def get_model_names(model):
    for name, layer in model.named_modules():
        print(name, layer)

def get_children(model):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def plot_model_info(model_info, title):
    colors = list(mcolors.CSS4_COLORS.keys())
    model_info_flatten = [item for sublist in model_info for item in sublist]
    x = [i for i in range(0, len(model_info_flatten))]
    h = model_info_flatten
    c = []
    for i in range(0, len(model_info)):
        for j in range(0, len(model_info[i])):
            c.append(str(colors[10 + i]))
    plt.figure()
    plt.bar(x, height = h, color = c)
    plt.title(title)
    plt.savefig(f'{title}.jpg')

parser = argparse.ArgumentParser(description='Get Model Information')
parser.add_argument('--config_path',
                        type = str,
                        default = '../first_order_model/config/api_sample.yaml',
                        help = 'path to the config file')


if __name__ == '__main__':
    args = parser.parse_args()
    model = FirstOrderModel(args.config_path)

    model_info = get_model_info(model.generator)
    print("Number of parameters in the generator", get_n_params(model.generator))
    plot_model_info(model_info, "Generator")

    model_info = get_model_info(model.kp_detector)
    print("Number of parameters in the kp detctor", get_n_params(model.kp_detector))
    plot_model_info(model_info, "KP Detector")



