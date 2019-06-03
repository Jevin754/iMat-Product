# iMat-Product
iMaterialist Challenge (FGVC6, 2019), 13th place

iMaterialist Challenge on Product Recognition (FGVC6, CVPR 2019): [kaggle page](https://www.kaggle.com/c/imaterialist-product-2019)


## Backbone
I use the following 5 CNN backbones pretrained on ImageNet. All pretrained models are from [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch).
* NASNet-A-Large
* SENet154
* InceptionResNet-v2
* ResNet152
* ResNet101

## Settings
* Learning rate: 4e-4
* Weight decay: 5e-5
* Epoch: all models converge in around 10 epochs

## Fusing
Nothing special, average of all probabilities.
