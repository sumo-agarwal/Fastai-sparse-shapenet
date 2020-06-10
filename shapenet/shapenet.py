import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys
import math
import numpy as np
import pandas as pd
import datetime
import glob
from IPython.display import display, HTML, FileLink
from os.path import join, exists, basename, splitext
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import cm
from joblib import cpu_count
from tqdm import tqdm
from fastai_sparse import utils
from fastai_sparse.utils import log, log_dict, print_random_states
from fastai_sparse.datasets import find_files, PointsDataset
from fastai_sparse import visualize
from datasets import DataSourceConfig, reader_fn
import transform as T
from fastai_sparse.data import SparseDataBunch
from data import merge_fn
from fastai_sparse.learner import SparseModelConfig, Learner
from torch import optim
from functools import partial
from fastai.callbacks.general_sched import TrainingPhase, GeneralScheduler
from fastai.callback import annealing_exp
from fastai_sparse.callbacks import TimeLogger, SaveModelCallback, CSVLogger, CSVLoggerIouByCategory
from metrics import IouByCategories
from fastai_sparse.core import num_cpus

experiment_name = 'shapenet'
SOURCE_DIR = Path('data').expanduser()
DIR_TRAIN_VAL = SOURCE_DIR / 'train_val'
categories = [
    "02691156", "02773838", "02954340", "02958343", "03001627", "03261776",
    "03467517", "03624134", "03636649", "03642806", "03790512", "03797390",
    "03948459", "04099429", "04225987", "04379243"
]

classes = [
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife',
    'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard',
    'Table'
]

num_classes_by_category = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
df_train = find_files(path=SOURCE_DIR / 'npy' / 'train', ext='.points.npy', ext_labels='.labels.npy', categories=categories)
df_valid = find_files(path=SOURCE_DIR / 'npy' / 'valid', ext='.points.npy', ext_labels='.labels.npy', categories=categories)
train_source_config = DataSourceConfig(                                       
                                       root_dir=SOURCE_DIR / 'npy' / 'train',                                       
                                       df=df_train,
                                       batch_size=16,
                                       num_workers=12,
                                       num_classes=50,
                                       num_classes_by_category=num_classes_by_category,
                                      )
train_source_config
valid_source_config = DataSourceConfig(                                       
                                       root_dir=SOURCE_DIR / 'npy' / 'valid',
                                       df=df_valid,
                                       batch_size=16,
                                       num_workers=12,
                                       num_classes=50,
                                       num_classes_by_category=num_classes_by_category,
                                       init_numpy_random_seed=False,
                                      )
valid_source_config
train_items = PointsDataset.from_source_config(train_source_config, reader_fn=reader_fn)
valid_items = PointsDataset.from_source_config(valid_source_config, reader_fn=reader_fn)
resolution = 24

PRE_TFMS = [
           T.fit_to_sphere(center=False),
           T.shift_labels(offset=-1)
           ]

AUGS_TRAIN = [
    T.rotate(),
    T.flip_x(p=0.5),
]

AUGS_VALID = [
    T.rotate(),
    T.flip_x(p=0.5),
]

SPARSE_TFMS = [
    T.translate(offset=2),
    T.scale(scale=resolution),
    T.merge_features(ones=True),
    T.to_sparse_voxels(),
]

tfms = (
    PRE_TFMS + AUGS_TRAIN + SPARSE_TFMS,
    PRE_TFMS + AUGS_VALID + SPARSE_TFMS,
)

train_items.transform(tfms[0])
pass

valid_items.transform(tfms[1])
pass
data = SparseDataBunch.create(train_ds=train_items,
                              valid_ds=valid_items,
                              collate_fn=merge_fn,)

model_config = SparseModelConfig(spatial_size=24 * 8, num_input_features=1)
model_config.check_accordance(data.train_ds.source_config, sparse_item=data.train_ds[0])
class Model(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        
        spatial_size = torch.LongTensor([cfg.spatial_size]*3)
        
        self.sparseModel = scn.Sequential(
            scn.InputLayer(cfg.dimension, spatial_size, mode=cfg.mode),
            scn.SubmanifoldConvolution(cfg.dimension, nIn=cfg.num_input_features, nOut=cfg.m, filter_size=3, bias=cfg.bias),
            scn.UNet(cfg.dimension, cfg.block_reps, cfg.num_planes, residual_blocks=cfg.residual_blocks, downsample=cfg.downsample),
            scn.BatchNormReLU(cfg.m),
            scn.OutputLayer(cfg.dimension),
        )
        self.linear = nn.Linear(cfg.m, cfg.num_classes)

    def forward(self, xb):
        coords = xb['coords']
        features = xb['features']
        x = [coords, features]

        x = self.sparseModel(x)
        x = self.linear(x)
        return x

model = Model(model_config)
learn = Learner(data, model,
                opt_func=partial(optim.SGD, momentum=0.9),
                wd=1e-4,
                true_wd=False,
                path=str(Path('results', experiment_name)))
learn.callbacks = []
cb_iou = IouByCategories(learn, len(categories))
learn.callbacks.append(cb_iou)

learn.callbacks.append(TimeLogger(learn))
learn.callbacks.append(CSVLogger(learn))
learn.callbacks.append(CSVLoggerIouByCategory(learn, cb_iou, categories_names=classes))
learn.callbacks.append(SaveModelCallback(learn, every='epoch', name='weights', overwrite=True))
learn.fit(10)
