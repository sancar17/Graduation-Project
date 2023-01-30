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

best_val = 1000

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
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

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

        x = self.final_conv(x)

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


def run_model(
    train_model: torch.nn.Module,
    dataset: T4CDataset,
    random_seed: int,
    train_fraction: float,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    epochs: int,
    dataloader_config: dict,
    optimizer_config: dict,
    log_file_name: str,
    device: str = None,
    limit: Optional[int] = None,
):  # noqa

    logging.info("dataset has size %s", len(dataset))
    data_parallel=False

    # Train / Dev / Test set splits
    logging.info("train/dev split")
    full_dataset_size = len(dataset)

    effective_dataset_size = full_dataset_size
    if limit is not None:
        effective_dataset_size = min(full_dataset_size, limit)
    indices = list(range(full_dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    assert np.isclose(train_fraction + val_fraction, 1.0)
    num_train_items = max(int(np.floor(train_fraction * effective_dataset_size)), batch_size)
    num_val_items = max(int(np.floor(val_fraction * effective_dataset_size)), batch_size)

    logging.info(
        "Taking %s from dataset of length %s, splitting into %s train items and %s val items",
        effective_dataset_size,
        full_dataset_size,
        num_train_items,
        num_val_items,
    )

    # Data loaders
    train_indices, dev_indices = indices[:num_train_items], indices[num_train_items : num_train_items + num_val_items]

    indices = list(range(full_dataset_size))

    
    train_sampler = SubsetRandomSampler(train_indices)
    dev_sampler = SubsetRandomSampler(dev_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, **dataloader_config)
    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=dev_sampler, **dataloader_config)

    # Optimizer
    if "lr" not in optimizer_config:
        optimizer_config["lr"] = 1e-4

    if device is None:
        logging.warning("device not set, torturing CPU.")
        device = "cpu"
        # TODO data parallelism and whitelist

    '''if torch.cuda.is_available() and data_parallel:
        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        if torch.cuda.device_count() > 1:
            # https://stackoverflow.com/questions/59249563/runtimeerror-module-must-have-its-parameters-and-buffers-on-device-cuda1-devi
            train_model = torch.nn.DataParallel(train_model)
            logging.info(f"Let's use {len(train_model.device_ids)} GPUs: {train_model.device_ids}!")
            device = f"cuda:{train_model.device_ids[0]}"'''

    optimizer = optim.Adam(train_model.parameters(), **optimizer_config)

    train_model = train_model.to(device)

    # Loss
    loss = F.mse_loss
    
    train_pure_torch(device, epochs, optimizer, train_loader, val_loader, train_model, log_file_name)

    logging.info("End training of train_model %s on %s for %s epochs", train_model, device, epochs)
    return train_model, device

def train_pure_torch(device, epochs, optimizer, train_loader, val_loader, train_model, log_file_name):
    best_loss = 10000

    for epoch in range(epochs):
        _train_epoch_pure_torch(train_loader, device, train_model, optimizer, log_file_name)
        loss = _val_pure_torch(val_loader, device, train_model, log_file_name, epoch)
        if loss<best_loss:
            best_loss = loss
            torch.save(train_model.state_dict(), "best3d_"+ log_file_name + ".pt")
        log = "Epoch: {:03d}, Test: {:.4f}"
        logging.info(log.format(epoch, loss))

        f = open("UNet3D_val_epoch_summary_" + log_file_name + ".txt", "a")
        f.write(str(loss) + "\n")
        f.close()

    torch.save(train_model.state_dict(), "final3d_" + log_file_name + ".pt")
        


def _train_epoch_pure_torch(loader, device, model, optimizer, log_file_name):
    loss_to_print = 0
    for i, input_data in enumerate(tqdm.tqdm(loader, desc="train")):
         
        input_data, ground_truth = input_data
        input_data = input_data.to(device)
        ground_truth = ground_truth.to(device)
    
        model.train()
        optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        output = model(input_data)
        output = output[:,:,(0,1,2,5,8,11),:,:]
        loss_speed = criterion(output[:,(0,2,4,6),:,:,:], ground_truth[:,(0,2,4,6),:,:,:])
        loss_volume = criterion(output[:,(1,3,5,7),:,:,:], ground_truth[:,(1,3,5,7),:,:,:])
        loss = 0.01*loss_speed + 0.99*loss_volume
        loss.backward()
        optimizer.step()

        loss_to_print += float(loss)
        if i % 1000 == 0 and i > 0:
            logging.info("train_loss %s", loss_to_print / 1000)
            loss_to_print = 0

    f = open("UNet3D_train_epoch_summary_" + log_file_name +".txt", "a")
    f.write(str(loss_to_print / len(loader)) + "\n")
    f.close()

@torch.no_grad()
def _val_pure_torch(loader, device, model, log_file_name, epoch):
    running_loss = 0
    for input_data in tqdm.tqdm(loader, desc="val"):
        input_data, ground_truth = input_data
        input_data = input_data.to(device)
        ground_truth = ground_truth.to(device)

        model.eval()
        criterion = torch.nn.MSELoss()
        output = model(input_data)
        output = output[:,:,(0,1,2,5,8,11),:,:]
        loss = criterion(output, ground_truth)
        running_loss = running_loss + float(loss)
    
    data = output.cpu()
    data = torch.reshape(data, (data.shape[0], data.shape[2], data.shape[3], data.shape[4], data.shape[1])) 
    
    if (epoch == 0):
        input_data = input_data.cpu()
        input_data = torch.reshape(input_data, (input_data.shape[0], input_data.shape[2], input_data.shape[3], input_data.shape[4], input_data.shape[1]))
        write_data_to_h5(input_data, "val_input_" + str(log_file_name) + ".h5")

    write_data_to_h5(data, "val_" + str(log_file_name) + "_epoch" + str(epoch) + ".h5")
    return running_loss / len(loader) if len(loader) > 0 else running_loss

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_checkpoint", type=str, help="torch pt file to be re-loaded.", default=None, required=False)
    parser.add_argument("--data_raw_path", type=str, help="Base dir of raw data", default="./data/raw")
    parser.add_argument("--train_fraction", type=float, default=0.9, required=False, help="Fraction of the data set for training.")
    parser.add_argument("--val_fraction", type=float, default=0.1, required=False, help="Fraction of the data set for validation.")
    parser.add_argument("--batch_size", type=int, default=1, required=False, help="Batch Size for training and validation.")
    parser.add_argument("--num_workers", type=int, default=2, required=False, help="Number of workers for data loader.")
    parser.add_argument("--epochs", type=int, default=200, required=False, help="Number of epochs to train.")
    parser.add_argument("--file_filter", type=str, default="**/*8ch.h5", required=False, help='Filter files in the dataset. Defaults to "**/*8ch.h5"')
    parser.add_argument("--limit", type=int, default=100, required= False, help="Cap dataset size at this limit.")
    parser.add_argument("--device", type=str, default="cuda:0", required=False, help="Force usage of device.")
    parser.add_argument("--device_ids", nargs="*", default=None, required=False, help="Whitelist of device ids. If not given, all device ids are taken.")
    parser.add_argument("--data_parallel", default=False, required=False, help="Use DataParallel.", action="store_true")
    parser.add_argument("--num_tests_per_file", default=100, type=int, required=False, help="Number of test slots per file")
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default="./",
        required=False,
        help='If given, submission is evaluated from ground truth zips "ground_truth_[spatio]tempmoral.zip" from this directory.',
    )
    parser.add_argument("--log_file_name", type=str, required=True)
    return parser

def main(args):

    t4c_apply_basic_logging_config()
    parser = create_parser()
    args = parser.parse_args(args)

    model_str = "UNet3D"
    resume_checkpoint = args.resume_checkpoint

    competitions = ["temporal", "spatiotemporal"]
    epochs = args.epochs
    train_fraction = args.train_fraction
    val_fraction = args.val_fraction
    ground_truth_dir = args.ground_truth_dir
    batch_size = args.batch_size
    random_seed = 123
    num_workers = args.num_workers
    num_tests_per_file = args.num_tests_per_file
    batch_size_scoring = 1
    limit = args.limit
    log_file_name = args.log_file_name

    device = args.device 
    full_handler = logging.FileHandler(model_str + "_log" + args.log_file_name + ".txt")
    full_handler.setLevel(logging.INFO)
    full_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s]%(message)s"))
    full_logger = logging.getLogger()
    full_logger.addHandler(full_handler)

    logging.info("Start build dataset")
    # Data set

    data_raw_path = args.data_raw_path
    file_filter = args.file_filter

    dataset_config = configs[model_str].get("dataset_config", {})

    dataset = T4CDataset(root_dir=data_raw_path, file_filter=file_filter, **dataset_config)
    logging.info("Dataset has size %s", len(dataset))
    assert len(dataset) > 0

    # Model

    logging.info("Getting model")
    model = UNet3D()
    
    dataloader_config = configs[model_str].get("dataloader_config", {})
    optimizer_config = configs[model_str].get("optimizer_config", {})

    if resume_checkpoint is not None:
        logging.info("Reload checkpoint %s", resume_checkpoint)
        load_torch_model_from_checkpoint(checkpoint=resume_checkpoint, model=model)

    logging.info("Going to run train_model.")
    logging.info(system_status())
    _, device = run_model(
        train_model=model, device = device, random_seed=random_seed, dataset=dataset, epochs=epochs, dataloader_config=dataloader_config, optimizer_config=optimizer_config, train_fraction=train_fraction, val_fraction=val_fraction, batch_size=batch_size, num_workers=num_workers, limit=limit, log_file_name=log_file_name)

    model.load_state_dict(torch.load("best3d_"+ log_file_name + ".pt"))
    
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


if __name__ == "__main__":
    main(sys.argv[1:])