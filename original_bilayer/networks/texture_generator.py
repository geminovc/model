# Third party
import torch
from torch import nn
import torch.nn.functional as F
import math

# This project
from original_bilayer.runners import utils as rn_utils
from original_bilayer.networks import utils as nt_utils



class NetworkWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        parser.add('--tex_num_channels',         default=64, type=int, 
                                                 help='minimum number of channels')

        parser.add('--tex_max_channels',         default=512, type=int, 
                                                 help='maximum number of channels')

        parser.add('--tex_norm_layer_type',      default='ada_spade_bn', type=str,
                                                 help='norm layer inside the texture generator')

        parser.add('--tex_pixelwise_bias_type',  default='none', type=str,
                                                 help='pixelwise bias type for convolutions')

        parser.add('--tex_input_tensor_size',    default=4, type=int, 
                                                 help='input spatial size of the generators')

        parser.add('--tex_activation_type',      default='leakyrelu', type=str,
                                                 help='activation layer inside the generators')

        parser.add('--tex_upsampling_type',      default='nearest', type=str,
                                                 help='upsampling layer inside the generator')

        parser.add('--tex_skip_layer_type',      default='ada_conv', type=str,
                                                 help='skip connection layer type')

        parser.add('--texture_output_dim',       default=3, type=int,
                                                 help='texture output dimensions, 3 for usual, 16 for unet added')
        # Dropout options
        parser.add('--use_dropout',              default='False', type=rn_utils.str2bool, choices=[True, False],
                                                 help='use dropout in the convolutional layers')

        parser.add('--dropout_networks',         default='texture_generator: 0.5' ,
                                                 help='networks to use dropout in: the dropout rate')
           
        # Replacement options
        parser.add('--replace_Gtex_output_with_trainable_tensor',   default='False', type=rn_utils.str2bool, choices=[True, False],
                                                                    help='set to true if you want to replace all of G_tex with a tensor')

        parser.add('--replace_Gtex_output_with_source',             default='False', type=rn_utils.str2bool, choices=[True, False],
                                                                    help='set to true if you want to replace all of G_tex output with source')

    def __init__(self, args):
        super(NetworkWrapper, self).__init__()
        # Initialize options
        self.args = args

        # Fully training the texture generator
        if not self.args.replace_Gtex_output_with_trainable_tensor:
            # Generator
            self.gen_tex_input = nn.Parameter(torch.randn(1, args.tex_max_channels, \
                    args.tex_input_tensor_size, args.tex_input_tensor_size))
            self.gen_tex = Generator(args)
            # Projector (prediction of adaptive parameters)
            self.prj_tex = Projector(args)

        # Replacing the texture generator with a trainable tensor
        else:
            self.gen_tex_output = nn.Parameter(torch.randn(1, 3, self.args.image_size , self.args.image_size))
        

    def forward(
            self, 
            data_dict: dict,
            networks_to_train: list,
            all_networks: dict, # dict of all networks in the model
        ) -> dict:

        # Do not store activations if this network is not being trained
        if 'texture_generator' not in networks_to_train:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        if not self.args.replace_Gtex_output_with_trainable_tensor and \
        not self.args.replace_Gtex_output_with_source:
            ### Prepare inputs ###
            idt_embeds = data_dict['source_idt_embeds']
            b = idt_embeds[0].shape[0]

            ### Forward through the projectors ###
            tex_weights, tex_biases = self.prj_tex(idt_embeds)
            self.assign_adaptive_params(self.gen_tex, tex_weights, tex_biases)

            ### Forward through the texture generator ###
            tex_inputs = torch.cat([self.gen_tex_input]*b, dim=0)
            outputs = self.gen_tex(tex_inputs)
            pred_tex_hf_rgbs = outputs[0]

            if 'texture_generator' not in networks_to_train:
                torch.set_grad_enabled(prev)

            ### Store outputs ###
            reshape_target_data = lambda data: data.view(b, t, *data.shape[1:])
            reshape_source_data = lambda data: data.view(b, n, *data.shape[1:])

            data_dict['pred_tex_hf_rgbs'] = pred_tex_hf_rgbs[:, None]
        
        elif self.args.replace_Gtex_output_with_trainable_tensor:
            data_dict['pred_tex_hf_rgbs']  = self.gen_tex_output [:, None]

        elif self.args.replace_Gtex_output_with_source:
            data_dict['pred_tex_hf_rgbs'] = data_dict['source_imgs']

        return data_dict

    @torch.no_grad()
    def visualize_outputs(self, data_dict):
        # All visualization is done in the inference generator
        visuals = []

        return visuals

    @staticmethod
    def assign_adaptive_params(net, weights, biases):
        i = 0
        for m in net.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d" and 'spade' not in m.norm_layer_type:
                m.weight = weights[i] + 1.0
                m.bias = biases[i]
                i += 1

            elif m.__class__.__name__ == 'AdaptiveConv2d':
                m.weight = weights[i]
                m.bias = biases[i]
                i += 1

    @staticmethod
    def adaptive_params_mixing(net, indices):
        for m in net.modules():
            if m.__class__.__name__ == "AdaptiveNorm2d" and 'spade' not in m.norm_layer_type:
                m.weight = m.weight[indices]
                m.bias = m.bias[indices]

            elif m.__class__.__name__ == 'AdaptiveConv2d':
                m.weight = m.weight[indices]
                m.bias = m.bias[indices]

    def __repr__(self):
        output = ''

        num_params = 0
        for p in self.prj_tex.parameters():
            num_params += p.numel()
        output += self.prj_tex.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params
        output += '\n'

        num_params = 0
        for p in self.gen_tex.parameters():
            num_params += p.numel()
        output += self.gen_tex.__repr__()
        output += '\n'
        output += 'Number of parameters: %d' % num_params

        return output


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        # Set options for the blocks
        num_blocks = int(math.log(args.image_size // args.tex_input_tensor_size, 2))
        
        out_channels = min(int(args.tex_num_channels * 2**num_blocks), args.tex_max_channels)
        spatial_size = 1

        if 'spade' in args.tex_norm_layer_type or args.tex_pixelwise_bias_type != 'none':
            spatial_size = args.tex_input_tensor_size

        layers = []

        # Construct the upsampling blocks
        for i in range(num_blocks - 1, -1, -1):
            in_channels = out_channels
            out_channels = min(int(args.tex_num_channels * 2**i), args.tex_max_channels)

            layers += [nt_utils.ResBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=2, 
                eps=args.eps,
                spatial_size=spatial_size,
                activation_type=args.tex_activation_type, 
                norm_layer_type=args.tex_norm_layer_type,
                resize_layer_type=args.tex_upsampling_type,
                pixelwise_bias_type=args.tex_pixelwise_bias_type,
                skip_layer_type=args.tex_skip_layer_type,
                first_norm_is_not_adaptive=i == num_blocks - 1)]

            if 'spade' in args.tex_norm_layer_type or args.tex_pixelwise_bias_type != 'none':
                spatial_size *= 2

        norm_layer = nt_utils.norm_layers[args.tex_norm_layer_type]
        activation = nt_utils.activations[args.tex_activation_type]

        layers += [
            norm_layer(out_channels, spatial_size, eps=args.eps),
            activation(inplace=True)]
        
        # Drop out layer before the final conv. layer
        nets_dropout = rn_utils.parse_str_to_dict(args.dropout_networks, value_type=float)
        if args.use_dropout and 'texture_generator' in nets_dropout.keys():
            dropout_rate = nets_dropout['texture_generator']
            layers += [nn.Dropout(p=dropout_rate, inplace=False)]

        self.blocks = nn.Sequential(*layers)

        # Get the list of required heads
        heads = [(args.texture_output_dim, nn.Tanh)]

        # Initialize the heads
        self.heads = nn.ModuleList()

        for num_outputs, final_activation in heads:
            layers = [nn.Conv2d(
                in_channels=out_channels, 
                out_channels=num_outputs, 
                kernel_size=3, 
                stride=1, 
                padding=1)]

            if final_activation is not None:
                layers += [final_activation()]

            self.heads += [nn.Sequential(*layers)]

    def forward(self, inputs):
        outputs = self.blocks(inputs).contiguous()

        results = []

        for head in self.heads:
            results += [head(outputs)]

        return results


class Projector(nn.Module):
    def __init__(self, args, bottleneck_size=1024):
        super(Projector, self).__init__()
        # Calculate parameters of the blocks
        num_blocks = int(math.log(args.image_size // args.tex_input_tensor_size, 2))
        
        # FC channels perform a lowrank matrix decomposition
        self.channel_mults = []
        self.avgpools = nn.ModuleList()
        self.fc_blocks = nn.ModuleList()

        for i in range(num_blocks, 0, -1):
            in_channels = min(int(args.emb_num_channels * 2**i), args.emb_max_channels)

            out_in_channels = min(int(args.tex_num_channels * 2**i), args.tex_max_channels)
            out_channels = min(int(args.tex_num_channels * 2**(i-1)), args.tex_max_channels)

            channel_mult = out_in_channels / float(in_channels)
            self.channel_mults += [channel_mult]

            # Average pooling is applied to embeddings before FC
            s = int(bottleneck_size**0.5 * channel_mult)
            self.avgpools += [nn.AdaptiveAvgPool2d((s, s))]

            # Define decompositions for the i-th block
            self.fc_blocks += [nn.ModuleList()]

            # First AdaptiveBias
            if args.tex_pixelwise_bias_type == 'adaptive':
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1))]

            # First AdaNorm or SPADE weights and biases
            if 'ada_spade' in args.tex_norm_layer_type:
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1)),
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1))]

            elif 'ada' in args.tex_norm_layer_type:
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, 2))]

            # Second AdaptiveBias
            if args.tex_pixelwise_bias_type == 'adaptive':
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1))]

            # Skip conv weights and biases
            if args.tex_skip_layer_type == 'ada_conv':
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_in_channels + 1))]

            # Second AdaNorm or SPADE weights and biases
            if 'ada_spade' in args.tex_norm_layer_type:
                self.fc_blocks[-1] += [
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1)),
                    nn.Sequential(
                        nn.Linear(int(s**2 / channel_mult), in_channels),
                        nn.Linear(in_channels, in_channels),
                        nn.Linear(in_channels, out_channels + 1))]

            elif 'ada' in args.tex_norm_layer_type:
                self.fc_blocks[-1] += [nn.Sequential(
                    nn.Linear(int(s**2 / channel_mult), in_channels),
                    nn.Linear(in_channels, in_channels),
                    nn.Linear(in_channels, 2))]

    def forward(self, embeds):
        weights = []
        biases = []

        for embed, fc_block, channel_mult, avgpool in zip(embeds, self.fc_blocks, self.channel_mults, self.avgpools):
            b, c, h, w = embed.shape
            embed = avgpool(embed)

            c_out = int(c * channel_mult)
            embed = embed.view(b * c_out, -1)

            for fc in fc_block:
                params = fc(embed)

                params = params.view(b, c_out, -1)
                
                weight = params[:, :, :-1].squeeze()

                if weight.shape[0] != b and len(weight.shape) > 1 or len(weight.shape) > 2:
                    weight = weight[..., None, None] # 1x1 conv weight

                bias = params[:, :, -1].squeeze()

                if b == 1:
                    weight = weight[None]
                    bias = bias[None]
                    
                weights += [weight]
                biases += [bias]

        return weights, biases
