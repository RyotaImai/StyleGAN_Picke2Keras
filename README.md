
# StyleBasedGAN with Keras
オリジナルのStyleGANのコード(https://github.com/NVlabs/stylegan) で学習したStyleBasedGANを扱いやすいKerasの形式に変換するリポジトリです。
現段階ではGeneratorとしての機能のみでDiscriminatorは作っていませんので学習機能は存在しません。
## Sample Weight
Please Download here
1024x1024_ffhq  
https://drive.google.com/file/d/1TV4TcgOQyqx8i7R6NVXHJWjcxVXFYgAT/view?usp=sharing
## 環境
Cuda 10.1  
tensorflow-gpu == 1.15  
Keras == 2.3.1  
この環境だとNvidiaのコードを動かすときにめちゃくちゃWarning出ますがとりあえず動きます。。
またオリジナルの環境が動く必要があります。

## 使い方
### PickelからKerasモデルへ変換
まず以下のpythonコードを実行しPickleからkerasのweightsであるh5ファイルへweightsを変換します。  
```
python Picke2Keras.py --SAVE_WEIGHTS_NAME="test.h5" --OUTPUT_RES=1024 --PICKLE_PATH="2019stylegan-ffhq-1024x1024.pkl"
```
### モデルへのweightsのロード
Keras_StyleGAN.pyはKerasで書き直したGeneratorが入っています。import後にStyleGANクラスを実行することでインスタンス化します。  
class内の各モデルが利用できます。
```
from Keras_StyleGAN import *
GAN = StyleGAN(output_res = 1024)
# synthesis_model
GAN.s_model()
# mapping_model
GAN.m_model()
# Combined_model
GAN.model()
```
現段階ではsynthesis_modelのみの移植が可能です。以下のコードでh5ファイルからKerasモデルにweightをロードしてください。
```
GAN.s_model.load_weights("test.h5")
```
### モデルの実行
synthesisモデルへの入力はinputとnoiseになります。noiseを入力するのが面倒な場合、GAN.Predict関数を用いると便利です。  
引数は以下の三つです。  
inputs    : 潜在ベクトルの入力  
noise     : ノイズ入力(設定は任意です。指定しない場合かつuse_noiseがTrueの場合内部で自動で生成されます。)  
network   : どのネットワークを用いるか(mapping / synthesisの二つから選んでください。)  
use_noise : ノイズ入力を使うかどうか。(Falseにする場合、ノイズの代わりにゼロパディングされた配列が入力されます。)  
```
imgs = GAN.Predict(dlatents)
```

## 残タスク
以下の課題はそのうちやります。。。
* Mappingの移植
* Discriminatorの移植
* Constantの改善
* use_wscaleなどオリジナルで実装されている学習のTipsの実装
* Trancation Trickの実装
