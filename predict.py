""" Runs text2mesh output in Cog"""
import argparse
import os
import random

# Run kaolin installation and setup
import subprocess
import sys
import time
from shutil import copyfile

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms
from PIL import Image

# os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

sys.path.insert(0, "./kaolin/")


import tempfile

# cog
from cog import BaseModel, BasePredictor, Input, Path

from main import run_branched


# Output
class Output(BaseModel):
    output_image: Path
    output_mesh: Path


# Instantiate Cog Predictor
class Predictor(BasePredictor):
    def setup(self):

        # Select torch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(
        self,
        source_mesh: str = Input(
            description="Source mesh",
            default="candle",
            choices=["alien", "candle", "horse", "lamp", "person", "shoe", "vase"],
        ),
        prompt: str = Input(
            description="Text input",
            default="an image of a candle made of colorful crochet",
        ),
        n_iter: int = Input(description="Number of iterations", default=750),
    ) -> Output:

        # base options
        args = set_default_options()

        # set custom options
        args.run = "branch"
        args.obj_path = f"data/source_meshes/{str(source_mesh)}.obj"
        args.output_dir = "results/"
        args.prompt = str(prompt)
        args.sigma = 5.0
        args.clamp = "tanh"
        args.n_normaugs = 4
        args.n_augs = 1
        args.normmincrop = 0.1
        args.normmaxcrop = 0.1
        args.geoloss = True
        args.colordepth = 2
        args.normdepth = 2
        args.frontview = True
        args.frontview_std = 4
        args.clipavg = "view"
        args.lr_decay = 0.9
        args.clamp = "tanh"
        args.normclamp = "tanh"
        args.maxcrop = 1.0
        args.save_render = True
        args.seed = 11
        args.n_iter = int(str(n_iter))
        args.learning_rate = 0.0005
        args.normal_learning_rate = 0.0005
        args.background = [1, 1, 1]
        args.frontview_center = [1.96349, 0.6283]

        # run model
        output_mesh_path = run_branched(args)

        # get last image
        last_image_path = os.path.join(args.output_dir, f"iter_{args.n_iter-100}.jpg")

        # save output image as Cog Path object
        return Output(
            output_image=Path(last_image_path), output_mesh=Path(output_mesh_path)
        )


# Args class
class Args:
    def __init__(self):
        pass


# Default options from main.py
def set_default_options():
    args = Args()

    args.obj_path = "meshes/mesh1.obj"
    args.prompt = "a pig with pants"
    args.normprompt = None
    args.promptlist = None
    args.normpromptlist = None
    args.image = None
    args.output_dir = "round2/alpha5"
    args.traintype = "shared"
    args.sigma = 10.0
    args.normsigma = 10.0
    args.depth = 4
    args.width = 256
    args.colordepth = 2
    args.normdepth = 2
    args.normwidth = 256
    args.learning_rate = 0.0005
    args.normal_learning_rate = 0.0005
    args.decay = 0
    args.lr_decay = 1
    args.lr_plateau = False
    args.pe = True
    args.decay_step = 100
    args.n_views = 5
    args.n_augs = 0
    args.n_normaugs = 0
    args.n_iter = 6000
    args.encoding = "gaussian"
    args.normencoding = "xyz"
    args.layernorm = False
    args.run = None
    args.gen = False
    args.clamp = "tanh"
    args.normclamp = "tanh"
    args.normratio = 0.1
    args.frontview = False
    args.no_prompt = False
    args.exclude = 0
    args.frontview_std = 8
    args.frontview_center = [0.0, 0.0]
    args.clipavg = None
    args.geoloss = False
    args.samplebary = False
    args.promptviews = None
    args.mincrop = 1
    args.maxcrop = 1
    args.normmincrop = 0.1
    args.normmaxcrop = 0.1
    args.splitnormloss = False
    args.splitcolorloss = False
    args.nonorm = False
    args.cropsteps = 0
    args.cropforward = False
    args.cropdecay = 1.0
    args.decayfreq = None
    args.overwrite = False
    args.show = False
    args.background = None
    args.seed = 0
    args.save_render = False
    args.input_normals = False
    args.symmetry = False
    args.only_z = False
    args.standardize = False
    return args
