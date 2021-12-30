from first_order_model.fom_wrapper import FirstOrderModel
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser(description='Get Model Information')
parser.add_argument('--config_path',
                        type = str,
                        default = '../first_order_model/config/api_sample.yaml',
                        help = 'path to the config file')


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
    moduels_children = []
    for module in modules:
        children_param = []
        children = get_children(module)
        moduels_children.append(children)
        try:
            for child in children:
                children_param.append(get_n_params(child))
        except:
            children_param.append(get_n_params(children))
        modules_params.append(children_param)
    return modules_params, modules, moduels_children


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


def process_model_info(model_info, modules, moduels_children, model_name):
    def show_child(i, j):
        try:
            print(moduels_children[i][j])
        except:
            print(moduels_children[i])

    def get_layers(min_th, max_th):
        for i in range(0, len(model_info)):
            module_info = model_info[i]
            for j in range(0, len(module_info)):
                layer_info = module_info[j]
                try:
                    if min_th < layer_info < max_th:
                        show_child(i, j)
                except:
                    pass

    print(f"# of params less than 2M in {model_name}")
    get_layers(0, 2000000)
    print(f"# of params greater than 2M, less than 8M in {model_name}")
    get_layers(2000000, 8000000)
    print(f"# of params greater than 8M in {model_name}")
    get_layers(8000000, 11000000)


if __name__ == '__main__':
    args = parser.parse_args()
    model = FirstOrderModel(args.config_path)

    model_info, modules, moduels_children = get_model_info(model.generator)
    process_model_info(model_info, modules, moduels_children, "Generator")
    print("Number of parameters in the generator", get_n_params(model.generator))
    plot_model_info(model_info, "Generator")

    model_info, modules, moduels_children = get_model_info(model.kp_detector)
    process_model_info(model_info, modules, moduels_children, "KP Detector")
    print("Number of parameters in the kp detctor", get_n_params(model.kp_detector))
    plot_model_info(model_info, "KP Detector")



