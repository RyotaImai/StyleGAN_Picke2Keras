import numpy as np
import keras
import tensorflow as tf 

from keras.layers import *
from keras.models import Model
import keras 
from keras.backend import tensorflow_backend as K

# Bias Add layer
class bias_add(Layer):
    def __init__(self, **kwargs):
        super(bias_add, self).__init__(**kwargs)

    def build(self,input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                      shape=(input_shape[3],),
                                      initializer=keras.initializers.Constant(value=0.5),
                                      trainable=True)
        self.bias = keras.backend.reshape(self.bias,(1,1,1,-1))
        super(bias_add, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.add(x,self.bias)
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
##-------------------------------- Style GAN ----------------------------------##
class StyleGAN():
    
    ##-------------------------------------------------------------------------##
    ##                            Config                                       ##
    ##-------------------------------------------------------------------------##

    def __init__(self,output_res=1024):
        self.resolution_log2   = int(np.log2(output_res))
        self.num_stages        = self.resolution_log2 - 1

        self.latent            = Input((2*self.num_stages,512))
        self.flat_Noise        = [Input((w,w,1)) for w in [2**(int(i/2)+2) for i in range(2*self.num_stages)]]
        self.Noise             = [
                                    [self.flat_Noise[2*i],self.flat_Noise[2*i+1]] for i in range(self.num_stages)
                                 ]
        
        self.s_model = self.synthesis_model()
        self.m_model = self.mapping_model()
        self.model   = self.mapping_synthesis_model()
    # 特徴マップの数を計算する関数
    # 引数のStageはBlockのことを指す。スタートは1から。出力が1024だとするとstageは[1,2,3,4,5,6,7,8,9]の9つとなる。
    def nf(self,stage): 
        fmap_decay        = 1
        fmap_max          = 512
        fmap_base         = 8192        
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    
    
    ##-------------------------------------------------------------------------##
    ##                            Blockを構成するレイヤー                        ##
    ##-------------------------------------------------------------------------##
    # 画像サイズをおおきくするレイヤー
    def _upscale2d(self,x,factor=2,gain=np.sqrt(2)):
        if gain != 1:
            x = Lambda(lambda x:x*gain)(x)
        if factor == 1:
            return x
        s = [0,int(x.shape[1]),int(x.shape[2]),int(x.shape[3])]
        x = Lambda(lambda x:tf.reshape(x,[-1,s[1],1,s[2],1,s[3]]))(x)
        x = Lambda(lambda x:tf.tile(x,[1,1,1,factor,1,factor]))(x)
        x = Lambda(lambda x:tf.reshape(x,[-1,s[1]*factor,s[2]*factor,s[3]]))(x)
        return x
    
    def UP_Conv(self,x,stage):
        if 2*int(x.shape[1]) < 128:
            x = Lambda(lambda x: self._upscale2d(x))(x)
            x = Convolution2D(self.nf(stage),(3,3),padding="same",use_bias=False,name="%ix%i/Conv0_up/weight"%(2**(stage+1),2**(stage+1)))(x)
        else:
            x = Conv2DTranspose(self.nf(stage),(4,4),padding="same",use_bias=False,strides=(2,2),name="%ix%i/Conv0_up/weight"%(2**(stage+1),2**(stage+1)))(x)
        return x
    
    
    # Blur カーネルの設定。Kerasレイヤーを用いているが本来は学習しないパラメータになるのでlambdaレイヤーで固定して用いる。
    def Blur(self,x):
        def blur_kernel(shape, dtype=np.float32):
            assert len(shape)==4
            f=[1,2,1]
            f = np.array(f, dtype=dtype)
            f = f[:, np.newaxis] * f[np.newaxis, :]
            f = f / np.sum(f)
            f = np.tile(f, [int(shape[2]),1, 1]).transpose((1,2,0))
            f = np.tile(f, [int(shape[3]),1 ,1, 1]).transpose((1,2,3,0))
            f = K.constant(f, dtype=dtype)
            return f

        def my_init(shape, dtype=None):
            return blur_kernel(shape,dtype=dtype)
        return DepthwiseConv2D(3, padding='same', depth_multiplier=1, use_bias=False, depthwise_initializer=my_init, trainable=False)(x)

    
    # Style Transfer を実装するレイヤー    
    def instance_norm(self,x, epsilon=1e-8):
        assert len(x.shape) == 4 # NCHW
        x -= tf.reduce_mean(x, axis=[1,2], keepdims=True) # 平均を0にする操作（AdaINレイヤに入れる前に必要）
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=[1,2], keepdims=True) + epsilon)
        return x

    
    # Block内のConvの後の処理をまとめたレイヤー
    def Layer_Epilogue(self,
        x,
        dlatent,
        stage,
        pre_or_post):
        assert pre_or_post in (0,1)
        
        if stage == 1:
            if pre_or_post:
                sub_stage = "Conv"
            else:
                sub_stage = "Const"
        else:
            if pre_or_post:
                sub_stage = "Conv1"
            else:
                sub_stage = "Conv0_up"
            
        noise = Dense(self.nf(stage),use_bias=False,name="%ix%i/%s/Noise/weight"%(2**(stage+1),2**(stage+1),sub_stage))(self.Noise[stage-1][pre_or_post])
        x = Add(name="%ix%i/%s/Noise_add"%(2**(stage+1),2**(stage+1),sub_stage))([x,noise])
        x = bias_add(name="%ix%i/%s/bias"%(2**(stage+1),2**(stage+1),sub_stage))(x)
        x = LeakyReLU(0.2)(x)
        x = Lambda(lambda x:self.instance_norm(x),name="%ix%i/%s/instance_norm"%(2**(stage+1),2**(stage+1),sub_stage))(x)
        x = self.style_mod(x,dlatent,stage,pre_or_post)  
        return x
    
    
    # Style Transfer を実装するレイヤー
    def style_mod(self,X,dlatent,stage,pre_or_post):
        # X : N,h,w,C dlatent : N,512
        if stage == 1:
            if pre_or_post:
                sub_stage = "Conv"
            else:
                sub_stage = "Const"
        else:
            if pre_or_post:
                sub_stage = "Conv1"
            else:
                sub_stage = "Conv0_up"        
        N,h,w,C = X.shape
        style = Dense(2*int(C),name="%ix%i/%s/StyleMod"%(2**(stage+1),2**(stage+1),sub_stage))(dlatent)
        style = Reshape((2,int(C)))(style)

        X = Multiply()([X,Lambda(lambda x: x[:,0]+1)(style)])
        X = Add()([X,Lambda(lambda x: x[:,1])(style)])
        return X

    ##-------------------------------------------------------------------------##
    ##                                Block                                    ##
    ##-------------------------------------------------------------------------##
    
    # First Block
    # Stage1を担当するBlock
    def First_Block(self,x,stage1_dlatents,stage=1):
        x = self.Layer_Epilogue(x,stage1_dlatents[0],stage,pre_or_post = 0)
        x = Convolution2D(self.nf(stage),(3,3),padding="same",use_bias=False,name="%ix%i/Conv"%(2**(stage+1),2**(stage+1)))(x)
        x = self.Layer_Epilogue(x,stage1_dlatents[1],stage,pre_or_post = 1)
        return x

    # Block
    # Stage2からの処理を担当する
    def Block(self,x,stageN_dlatents,stage):
        x = self.UP_Conv(x,stage)
        x = Lambda(lambda x:self.Blur(x),name = "%ix%i/Blur"%(2**(stage+1),2**(stage+1)))(x)
        x = self.Layer_Epilogue(x,stageN_dlatents[0],stage,pre_or_post = 0)
        x = Convolution2D(self.nf(stage),(3,3),padding="same",use_bias=False,name = "%ix%i/Conv1"%(2**(stage+1),2**(stage+1)))(x)
        x = self.Layer_Epilogue(x,stageN_dlatents[1],stage,pre_or_post = 1)
        return x
    
    ##-------------------------------------------------------------------------##
    ##                             その他のレイヤー                             ##
    ##-------------------------------------------------------------------------##
    
    def to_RGB(self,x,stage):
        x = Convolution2D(3,(1,1),name="ToRGB")(x)
        return x

    
    ##-------------------------------------------------------------------------##
    ##                             Sub Model                                   ##
    ##                        (mapping & synthesis)                            ##
    ##-------------------------------------------------------------------------##

    # mapping network
    def mapping(self,z):
        z = LeakyReLU(0.2)(Dense(512)(z))
        z = LeakyReLU(0.2)(Dense(512)(z))
        z = LeakyReLU(0.2)(Dense(512)(z))
        z = LeakyReLU(0.2)(Dense(512)(z))
        z = LeakyReLU(0.2)(Dense(512)(z))
        z = LeakyReLU(0.2)(Dense(512)(z))
        z = LeakyReLU(0.2)(Dense(512)(z))
        dlatent = LeakyReLU(0.2)(Dense(512)(z))
        return dlatent

    # synthesis network
    def synthesis(self,dlatent):
        ## dlatentsをリストの形式に変換する。
        dlatent_List = [[Lambda(lambda x:x[:,2*s])(dlatent),Lambda(lambda x:x[:,2*s+1])(dlatent) ] for s in range(self.num_stages)]
        # Constant Inputを作成
        constant = Dense(4*4*512,use_bias=True,name="4x4/Const/const",kernel_initializer="zeros")(Lambda(lambda x:(0*x[:,0:1]))(dlatent_List[0][0]))
        constant = Reshape((4,4,512),name="Constant_Reshape")(constant)
        x = self.First_Block(constant,dlatent_List[0])
        for i in range(1,self.num_stages):
            x = self.Block(x,dlatent_List[i],stage = i+1)
        img = self.to_RGB(x,stage = i+1)
        return img
    
    ##-------------------------------------------------------------------------##
    ##                                Model                                    ##
    ##-------------------------------------------------------------------------##
    

    def mapping_synthesis_model(self):
        self.dlatent = self.mapping(self.latent)
        img = self.synthesis(self.dlatent)
        return Model([self.latent]+self.flat_Noise,img)

    def mapping_model(self):
        self.dlatent = self.mapping(self.latent)
        return Model(self.latent,self.dlatent)
    
    def synthesis_model(self,fmt="Min_Max"):
        assert fmt in ("Min_Max","original")
        w = Input((2*self.num_stages,512))
        img = self.synthesis(w)
        if fmt == "original":
            return Model([w]+self.flat_Noise,img)
        elif fmt == "Min_Max":
            img = Lambda(lambda x:x*255/2+(0.5+255/2))(img)
            return Model([w]+self.flat_Noise,img)
        
    

    ##-------------------------------------------------------------------------##
    ##                                function                                 ##
    ##-------------------------------------------------------------------------##
    def Predict(self,inputs,noise=None,network = "synthesis",use_noise = True):
        assert network in ("mapping","synthesis","Comibined"),"Please select network type in mapping or synthesis"
        if noise != None:
            assert type(noise) == list, "noise is list type"
        if network == "mapping":
            return self.m_model.predict(inputs)

        elif network == "synthesis":
            length = len(inputs)
            if noise != None:
                return self.s_model.predict([inputs]+noise)
            elif use_noise:
                noise = [np.random.normal(size=(length,w,w,1)) for w in [2**(int(i/2)+2) for i in range(2*self.num_stages)]]
                return self.s_model.predict([inputs]+noise)
            else:
                noise = [np.zeros((length,w,w,1)) for w in [2**(int(i/2)+2) for i in range(2*self.num_stages)]]
                return self.s_model.predict([inputs]+noise)