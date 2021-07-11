## MILA segmentation challenge

## Implementation details

### Model details:
*  In this work, I employ 3 different widely known semantic segmentation architectures and 3 different backbones. In total, I report results for 8 different models, each of them trained from scratch. The 8 model variants are,

    1. [FCN](https://arxiv.org/abs/1411.4038) + [ResNet50](https://arxiv.org/abs/1512.03385) 
    2. [FCN](https://arxiv.org/abs/1411.4038) + [ResNet101](https://arxiv.org/abs/1512.03385)
    3. [Deeplabv3](https://arxiv.org/abs/1706.05587) + [ResNet50](https://arxiv.org/abs/1512.03385) 
    4. [Deeplabv3](https://arxiv.org/abs/1706.05587) + [ResNet101](https://arxiv.org/abs/1512.03385) 
    5. [Deeplabv3](https://arxiv.org/abs/1706.05587) + [MobileNetV3_large](https://arxiv.org/abs/1905.02244) 
    6. [Deeplabv3](https://arxiv.org/abs/1706.05587) + [MobileNetV3_small](https://arxiv.org/abs/1905.02244) 
    7. [Lite R-ASPP + MobileNetV3_large](https://arxiv.org/abs/1905.02244)  
    8. [Lite R-ASPP + MobileNetV3_small](https://arxiv.org/abs/1905.02244)  

*  Each model takes 3 channel imput. Output is a single channel 

### Training details:
*  All the images are standardized using normal standardization to have pixel values between 0 and 1. 
*  Each model is initialized with pretrained weights/pretrained backbone, and trained for maximum 30 epochs, with early stopping criteria.
*  Early stopping regularization is decided based on validation metric, which could either be weighted binary cross-entropy loss or IoU loss
*  Patience period of 10 epochs is used for early stopping. 
*  Here, I play with two different loss functions. Weighted binary cross-entropy or IoU loss(Jaccard index). Weighted BCE required tuning of weight hyperparameter for minoroty class
*  Reference papers:

    1. [Deeplabv3](https://arxiv.org/abs/1706.05587)
    2. [Lite R-ASPP](https://arxiv.org/abs/1905.02244)  
    3. [ResNet](https://arxiv.org/abs/1512.03385)
    
*  This work is fully implemented in pytorch. Following are the reference implementations used/adopted in this work,

    1. [Pytorch vision](https://github.com/pytorch/vision/tree/master/torchvision/models/segmentation) - used for Deeplabv3, ResNets, MobileNets
    2. [Early stopping](https://github.com/Bjarten/early-stopping-pytorch) - To implement early stopping
    3. [detectron2](https://github.com/facebookresearch/detectron2) - For config nodes
    
### Installing dependencies:

```
conda create -n mila-challenge python=3.8
conda activate mila-challenge
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge tensorboard matplotlib fvcore
conda install -c anaconda scikit-learn
pip install iopath gdown
```

### cloning the repository

```
git clone https://github.com/dhaivat1729/mila-AI-challenge.git
cd mila-AI-challenge
```

### Downloading models
```
gdown --id 1sD-2kBiAw94rwTCFi2xVP6YciqLvUm0D
unzip mila-segmentation-logs-20210711T195658Z-001.zip && rm mila-segmentation-logs-20210711T195658Z-001.zip 
```

## Expected structure for pretrained model directory:
```
mila-AI-challenge/
  mila-segmentation-logs/
    deeplabv3_mobilenet_v3_large_v2_no_L2_val_loss_metric/
    deeplabv3_mobilenet_v3_large_v3_jaccard_training/
    .
    .
    .
    .
    lraspp_mobilenet_v3_small_v3_jaccard_training/
```

### Training a model
```
python train_net.py -dataset_path '/path/to/segmentation_project/' -model_name <model name> -model_ver v1
```
* `<model name>` could be `fcn_resnet101`, `fcn_resnet50`, `deeplabv3_resnet50`, `deeplabv3_resnet101`, `lraspp_mobilenet_v3_large`, `deeplabv3_mobilenet_v3_large`, `deeplabv3_mobilenet_v3_small`, `lraspp_mobilenet_v3_small`
* `model_ver` is used to version the model incase multiple models of same architectures need to be trained.

## Expected structure for `dataset_path` directory:
```
segmentation_project/
  train/
    img/
      1.jpg
      2.jpg
      .
      .
      n.jpg
    mask/
      1.BMP
      2.BMP
      .
      .
      n.BMP
```

### Example:

#### If you need to train `lraspp_mobilenet_v3_large` model, then run the following command

```
python train_net.py -dataset_path '/path/to/segmentation_project/' -model_name lraspp_mobilenet_v3_large -model_ver v1
```

#### There are options to set batchsize, loss function, loss weights etc in `src/config/default.py`, feel free to change it there or overwrite it in `setup()` function in `utils.py`.

### Inference:
```
python infer.py /path/to/test/data/ /path/to/output/data/
```

### Example:

#### If we have images in `test_data/img` directory in this repository as below,

```
mila-AI-challenge/
  infer.py
  test_data/
    img/
      1.jpg
      2.jpg
        .
        .
      n.jpg
```

#### then we can run inference as below,

```
python infer.py test_data/img test_data/bmp_results
```

#### now there will be a new directory with results as below,

```
mila-AI-challenge/
  infer.py
  test_data/
    img/
      1.jpg
      2.jpg
        .
        .
      n.jpg
    bmp_results/
      1.BMP
      2.BMP
        .
        .
      n.BMP
```

#### You can run inference on any of the trained models by changing ` "model_directory" ` variable in `infer.py`
