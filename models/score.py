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

model_str = "unet"


full_handler = logging.FileHandler(model_str + "_score_log.txt")
full_handler.setLevel(logging.INFO)
full_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s]%(message)s"))
full_logger = logging.getLogger()
full_logger.addHandler(full_handler)

logging.info("Create train_model.")
model_class = configs[model_str]["model_class"]
model_config = configs[model_str].get("model_config", {})
model = model_class(**model_config)

competitions = ["temporal"]

data_raw_path = "../../NeurIPS2021-traffic4cast/data/raw"
model.load_state_dict(torch.load("bestunet.pt"))

device = "cuda:1"


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
        batch_size=1,
        num_tests_per_file=100,
        **additional_args,
    )