import argparse
import os
import random
from shutil import copyfile

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms
from PIL import Image


import os
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import subprocess
# subprocess.call(["git","clone", "--recursive", "https://github.com/NVIDIAGameWorks/kaolin"])
# subprocess.call(["python", "setup.py", "develop"], shell=True, cwd='./kaolin/')
# subprocess.call('ls', shell=True, cwd='../')

# cog
from cog import BasePredictor, Input, Path

import tempfile

from main import run_branched

class Args():
    def __init__(self):
        pass
# Instantiate Cog Predictor
class Predictor(BasePredictor):
    def setup(self):

        # Select torch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, 
        prompt: str = Input(description="Text input", default="an image of an alien made of cobblestone"),
        ) -> Path:

        

        args = Args()


        args.obj_path='meshes/mesh1.obj'
        args.prompt='a pig with pants'
        args.normprompt=None
        args.promptlist=None
        args.normpromptlist=None
        args.image=None
        args.output_dir='round2/alpha5'
        args.traintype='shared'
        args.sigma=10.0
        args.normsigma=10.0
        args.depth=4
        args.width=256
        args.colordepth=2
        args.normdepth=2
        args.normwidth=256
        args.learning_rate=0.0005
        args.normal_learning_rate=0.0005
        args.decay=0
        args.lr_decay=1
        args.lr_plateau=False
        args.pe=True
        args.decay_step=100
        args.n_views=5
        args.n_augs=0
        args.n_normaugs=0
        args.n_iter=6000
        args.encoding='gaussian'
        args.normencoding='xyz'
        args.layernorm=False
        args.run=None
        args.gen=False
        args.clamp='tanh'
        args.normclamp='tanh'
        args.normratio=0.1
        args.frontview=False
        args.no_prompt=False
        args.exclude=0
        args.frontview_std=8
        args.frontview_center=[0.0, 0.0]
        args.clipavg=None
        args.geoloss=False
        args.samplebary=False
        args.promptviews=None
        args.mincrop=1
        args.maxcrop=1
        args.normmincrop=0.1
        args.normmaxcrop=0.1
        args.splitnormloss=False
        args.splitcolorloss=False
        args.nonorm=False
        args.cropsteps=0
        args.cropforward=False
        args.cropdecay=1.0
        args.decayfreq=None
        args.overwrite=False
        args.show=False
        args.background=None
        args.seed=0
        args.save_render=False
        args.input_normals=False
        args.symmetry=False
        args.only_z=False
        args.standardize=False

        # custom 


        args.sigma = 5.0 
        args.clamp = "tanh"
        args.n_normaugs = 4
        args.n_augs = 1 
        args.normmincrop = 0.1 
        args.normmaxcrop =0.1 
        args.geoloss = True 
        args.colordepth = 2 
        args.normdepth = 2 
        # args.frontview = True 
        args.frontview_std = 4 
        args.clipavg = "view"
        args.lr_decay = 0.9
        args.normclamp = "tanh"  
        args.maxcrop = 1.0 
        args.save_render = True 
        args.seed = 41 
        # args.n_iter = 1500 
        args.learning_rate = 0.0005 
        args.normal_learning_rate = 0.0005  
        args.background = [1, 1, 1]
        args.frontview_center = [1.96349, 0.6283]

        args.obj_path = "data/source_meshes/person.obj"  #@param {type: "string"}
        args.n_iter = 750  #@param {type: "integer"}
        args.output_dir = "./results2"
        # args.prompt = "a 3D rendering of a ninja in unreal engine"
        args.run = "branch"
        args.prompt = prompt
        # python main.py --run branch 
        # --obj_path data/source_meshes/alien.obj 
        # --output_dir results/demo/alien/cobblestone 
        # --prompt an image of an alien made of cobblestone
        #  --sigma 5.0  --clamp tanh --n_normaugs 4 --n_augs 1 --normmincrop 0.1 --normmaxcrop 0.1 --geoloss --colordepth 2 --normdepth 2 --frontview --frontview_std 4 --clipavg view --lr_decay 0.9 --clamp tanh --normclamp tanh  --maxcrop 1.0 --save_render --seed 41 --n_iter 1500  --learning_rate 0.0005 --normal_learning_rate 0.0005  --background 1 1 1 --frontview_center 1.96349 0.6283


        run_branched(args)
        # #@export the results
        # import matplotlib.pyplot as plt
        # import importlib
        # import PIL
        # importlib.reload(PIL.TiffTags)
        # import cv2
        # import os


        # frames = []
        # for i in range(0, args.n_iter, 100):
        #     img = cv2.imread(os.path.join(args.output_dir, f"iter_{i}.jpg"))
        #     frames.append(img)
        #     plt.figure(figsize=(20, 4))
        #     plt.axis("off")
        #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     plt.show()
        
        # main(args) 

        last_img_path = os.path.join(args.output_dir, f"iter_{args.n_iter-100}.jpg")

        # save output image as Cog Path object
        output_path = Path(tempfile.mkdtemp()) / last_img_path
        output_model.save(output_path)
        print(output_path)
        return output_path
