import imageio
import torch
import numpy as np
import flow_vis
import matplotlib
import matplotlib.pyplot as plt
from skimage import img_as_float32
from first_order_model.modules.ifnet import *
from first_order_model.modules.warplayer import *


source_file = '/video-conf/scratch/vibhaa_tardy/dataset_256/xiran_video1_frames/test/frame_0001.png'
target_file = '/video-conf/scratch/vibhaa_tardy/dataset_256/xiran_video1_frames/test/frame_0011.png'


def read_single_frame(filename):
    """ read a single png file into a numpy array """
    image = imageio.imread(filename)
    return img_as_float32(image)


def frame_to_tensor(frame):
    """ convert numpy arrays to tensors for reconstruction pipeline """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    array = np.expand_dims(frame, 0).transpose(0, 3, 1, 2)
    array = torch.from_numpy(array)
    return array.float().to(device)


def draw_deformation_heatmap(deformation, filename):
    b, h, w = deformation.shape[0:3]
    identity_grid = np.zeros((h, w, 2))
    for i, ival in enumerate(np.linspace(-1, 1, h)):
        for j, jval in enumerate(np.linspace(-1, 1, w)):
            identity_grid[i][j][0] = jval
            identity_grid[i][j][1] = ival

    deformation_heatmap = np.zeros((b, h, w, 3))
    for i in range(b):
        deformation_heatmap[i] = flow_vis.flow_to_color(deformation[i] - identity_grid)
    plt.imsave(filename, deformation_heatmap[0])


def example_flow(source_file, target_file):
    source = frame_to_tensor(read_single_frame(source_file))
    target = frame_to_tensor(read_single_frame(target_file))

    # get optical flow from RIFE
    flownet = IFNet()
    flownet.cuda()
    scale_list = [4, 2, 1]
    timestep = 1
    imgs = torch.cat((source, target), 1)
    flow, mask, merged, flow_teacher, merged_teacher, loss_distill = flownet(imgs, scale_list, timestep=timestep)
    
    optical_flow = flow #merged[0]
    scale_to_select = 1
    src_deformation = get_warp_grid(source, optical_flow[scale_to_select][:, :2]).data.cpu().numpy()
    draw_deformation_heatmap(src_deformation, "src_flow.pdf")
    tgt_deformation = get_warp_grid(target, optical_flow[scale_to_select][:, 2:4]).data.cpu().numpy()
    draw_deformation_heatmap(tgt_deformation, "tgt_flow.pdf")
    """

    of_np = optical_flow.data.cpu().numpy()
    of_np = of_np.transpose(0, 2, 3, 1)[0]
    plt.imsave("flow.pdf", of_np)
    """
    #print(optical_flow.shape)
    #visual = flow_vis.flow_to_color(optical_flow_np)


example_flow(source_file, target_file)
