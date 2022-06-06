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


# cog
from cog import BasePredictor, Input, Path

import tempfile

from main import run_branched, make_parser

# Instantiate Cog Predictor
class Predictor(BasePredictor):
    def setup(self):

        # Select torch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, 
        prompt: str = Input(description="Text input", default="an image of an alien made of cobblestone"),
        ) -> Path:

        args = make_parser()
        args.run = "branch"
        # args.obj_path = "data/source_meshes/alien.obj"
        # args.output_dir = "results/demo/alien/cobblestone"
        args.prompt = prompt
        
        args.sigma = 5.0 
        args.clamp = "tanh"
        args.n_normaugs = 4
        args.n_augs = 1 
        args.normmincrop = 0.1 
        args.normmaxcrop =0.1 
        args.geoloss = True 
        args.colordepth = 2 
        args.normdepth = 2 
        args.frontview - True 
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
        args.prompt = "a 3D rendering of a ninja in unreal engine"

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