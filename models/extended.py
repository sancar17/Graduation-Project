import argparse
import binascii
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa
import torch.optim as optim
import torch_geometric
import tqdm
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import GradsHistHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import Checkpoint
from ignite.handlers import DiskSaver
from ignite.handlers import global_step_from_engine
from ignite.metrics import Loss
from ignite.metrics import RunningAverage
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from baselines.baselines_configs import configs
from baselines.checkpointing import load_torch_model_from_checkpoint
from baselines.checkpointing import save_torch_model_to_checkpoint
from competition.scorecomp import scorecomp
from competition.submission.submission import package_submission
from data.dataset.dataset import T4CDataset
from data.dataset.dataset_geometric import GraphTransformer
from data.dataset.dataset_geometric import T4CGeometricDataset
from util.logging import t4c_apply_basic_logging_config
from util.monitoring import system_status
from util.tar_util import untar_files
import torch.nn as nn
from util.h5_util import write_data_to_h5
from itertools import accumulate

from baselines.buildingblocks import DoubleConv, ExtResNetBlock, create_encoders, \
    create_decoders

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels = 8, out_channels = 8, basic_module=DoubleConv, f_maps=16, layer_order='cr',
                 num_groups=4, num_levels=2, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, **kwargs):
        super(Abstract3DUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output   
        # channels to the number of labels
        self.final_conv = nn.Conv3d(4, 4, 1)

        self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)


        x1 = self.final_conv(x[:,(0,2,4,6),:,:,:])
        x2 = self.final_conv(x[:,(1,3,5,7),:,:,:])
        
                
        x = torch.cat((x1[:,0,:,:,:], x2[:,0,:,:,:], x1[:,1,:,:,:], x2[:,1,:,:,:], x1[:,2,:,:,:], x2[:,2,:,:,:], x1[:,3,:,:,:], x2[:,3,:,:,:]),0)
        x = torch.unsqueeze(x,0)

        
        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels = 8, out_channels = 8, final_sigmoid=True, f_maps=16, layer_order='cr',
                 num_groups=4, num_levels=2, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     conv_padding=conv_padding,
                                     **kwargs)

competitions = ["spatiotemporal"]
data_raw_path = "../../NeurIPS2021-traffic4cast/data/raw"
model = UNet3D()
model.load_state_dict(torch.load("./best3d_mt_yarisma1.pt"))
model_str = "UNet3D"
device = "cuda:2"
batch_size = 1
num_tests_per_file = 100
ground_truth_dir = None

for competition in competitions:
        additional_args = {}
        submission = package_submission(
            data_raw_path=data_raw_path,
            competition=competition,
            model=model,
            model_str=model_str,
            device=device,
            h5_compression_params={"compression_level": None},
            submission_output_dir=Path("."),
            # batch mode for submission
            batch_size=batch_size,
            num_tests_per_file=num_tests_per_file,
            **additional_args,
        )
        ground_truth_dir = ground_truth_dir
        if ground_truth_dir is not None:
            ground_truth_dir = Path(ground_truth_dir)
            scorecomp.score_participant(
                ground_truth_archive=str(ground_truth_dir / f"ground_truth_{competition}.zip"),
                input_archive=str(submission)
            )
        else:
            scorecomp.verify_submission(input_archive=submission, competition=competition)
