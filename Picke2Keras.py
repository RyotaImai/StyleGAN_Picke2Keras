import numpy as np
import pandas as pd
from Keras_StyleGAN import *
from utils import *
from dnnlib.tflib import init_tf
import dnnlib.tflib
import h5py
import tqdm
import pickle
import argparse
init_tf()

parser = argparse.ArgumentParser(description='Pickle2Keras')
parser.add_argument('--PICKLE_PATH', default='', help='PICKLE_PATH')
parser.add_argument('--SAVE_WEIGHTS_NAME', default='', help='SAVE_WEIGHTS_NAME')
parser.add_argument('--OUTPUT_RES', default='', help='output resolution')
args = parser.parse_args()


num_stage = int(np.log2(int(args.OUTPUT_RES))-1)
df = pd.read_csv("converter.csv").iloc[:2+num_stage*10]
function_dict = {"Normal":Normal,"constant_reshape":constant_reshape,"axis_add":axis_add,"Upscale_conv":Upscale_conv,"conv":conv,"ToRGB":ToRGB,"StyleMod":StyleMod}

with open(args.PICKLE_PATH,"rb") as f:
    _G, _D, Gs = pickle.load(f);
    
GAN = StyleGAN(output_res=int(args.OUTPUT_RES))
layer_names = [GAN.s_model.layers[i].name for i in range(len(GAN.s_model.layers))]

for i in tqdm.tqdm(range(len(df))):
    name,index,original,func_name = df.iloc[i]
    weights = function_dict[func_name](Gs.components.synthesis.get_var(original))
    GAN.s_model.layers[layer_names.index(name)].weights[index].load(weights)

GAN.s_model.save_weights(args.SAVE_WEIGHTS_NAME)
