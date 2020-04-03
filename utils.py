import numpy as np
def Normal(x):
    return x

def constant_reshape(constant):
    constant = constant.transpose(0,2,3,1)
    return np.reshape(constant,(8192))

def axis_add(weight):
    return weight[np.newaxis]

def conv(kernel):
    h,w,In,Out = kernel.shape
    return kernel * np.sqrt(2) / np.sqrt(h*w*In)

def Upscale_conv(kernel):
    kernel = conv(kernel)
    new = kernel.transpose(0,1,3,2)
    new = np.pad(new,[[1,1], [1,1], [0,0], [0,0]],"constant")
    new = (new[1:,1:] + new[:-1,1:] + new[1:,:-1] + new[:-1,:-1])
    return new

def ToRGB(kernel):
    h,w,In,Out = kernel.shape
    return kernel / np.sqrt(h*w*In) # gain==1

def StyleMod(latents):
    In,Out = latents.shape # always 512
    return latents / np.sqrt(In)
