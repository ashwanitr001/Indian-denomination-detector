# Indian-denomination-detector using Keras and Tensorflow
Classification of Indian Currency to its respective values.
Currently the Model is trained on Rs.10, Rs.20, Rs.50, Rs.100, Rs.200, Rs.500 and  Rs.2000 notes only.

This repository contains implementation for multiclass image classification using Keras as well as Tensorflow.
To run the code:
* Set up the path to train and test directory in the code.
* Set up the parameters for model training.
* Select the model you want to use.(Currently the code supports: InceptionResNetV2, DenseNet121, ResNet50, NASNetMobile, MobileNet and InceptionV3).
* Generally the Nasnet is the most accurate but on the slower side as well as heavier. But for simple problems Resnet50 will do suffice.
* If processing is not the issue use NASNET as it'll give best results.
* A brief comparison of different models is given below:

## Comparison Table
CNN model comparison table on the [ImageNet](http://www.image-net.org/) classification results, reference paper and implementations.

Model | Detail | Input size | Parameters | Mult-Adds | FLOPS | Depth | Top-1 Acc | Top-5 Acc | Paper | TF | Keras | Pytorch | Caffe | Torch | MXNet
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
NASNet-A_Large_331 | (N=6, F=168) | 331x331⁵ | 88.9M⁵ | 23.8B⁵ |   |   | 82.70⁵ | 96.20⁵ | [Paper](https://arxiv.org/abs/1707.07012) | [TF](https://github.com/tensorflow/models/tree/master/research/slim) | [Keras](https://keras.io/applications/) | [Pytorch](https://github.com/wandering007/nasnet-pytorch) | - | - | -
Inception-ResNet-v2 |   | 299x299¹ | 55.8M² |   | 11.75G⁴ | 572² | 80.40¹ | 95.30¹ | [Paper](http://arxiv.org/abs/1602.07261) | [TF](https://github.com/tensorflow/models/tree/master/research/slim) | [Keras](https://keras.io/applications/) | - | [Caffe](https://github.com/twtygqyy/Inception-resnet-v2) | - | -
Inception V3 |   | 299x299¹ | 23.8M² |   |   | 159² | 78.00¹ | 93.90¹ | [Paper](http://arxiv.org/abs/1512.00567) | [TF](https://github.com/tensorflow/models/tree/master/research/slim) | [Keras](https://keras.io/applications/) | [Pytorch](https://github.com/pytorch/vision/tree/master/torchvision) | [Caffe](https://github.com/smichalowski/google_inception_v3_for_caffe) | - | -
ResNeXt50 | (32x4d) | 224x224³ | 25M³ |   | 3.768G³ |   | 77.15³ | 94.25³ | [Paper](https://arxiv.org/abs/1611.05431) | [TF](https://github.com/taki0112/ResNeXt-Tensorflow) | [Keras](https://github.com/titu1994/Keras-ResNeXt) | [Pytorch](https://github.com/prlz77/ResNeXt.pytorch) | [Caffe](https://github.com/cypw/ResNeXt-1) | [Torch](https://github.com/facebookresearch/ResNeXt) | -
DenseNet121 |   | 224x224² | 8M² |   |   | 121² | 74.50² | 91.80² | [Paper](https://arxiv.org/abs/1608.06993) | [TF](https://github.com/YixuanLi/densenet-tensorflow) | [Keras](https://keras.io/applications/) | [Pytorch](https://github.com/pytorch/vision/tree/master/torchvision) | [Caffe](https://github.com/shicai/DenseNet-Caffe) | - | -
Inception V2 |   | 224x224¹ |   |   |   |   | 73.90¹ | 91.80¹ | [Paper](http://arxiv.org/abs/1502.03167) | [TF](https://github.com/tensorflow/models/tree/master/research/slim) | - | [Pytorch](https://github.com/pytorch/vision/tree/master/torchvision) | - | - | -
