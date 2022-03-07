# DETR-pytorch object detection 

## Environment
* Ubuntu 20.04
* Nvidia TITAN Xp
* CUDA 11.3
* CUDNN 8.2.1
* torch 1.10.0
* python 3.8

## Backbone
* [x] ResNet50
* [x] MobileNetV2
* [x] MobileNetV3
* [x] ConvNeXt

## Train
```python
python main.py --coco_path 'path to coco' --backbone 'choose your backbone'
```

## todo 
* [ ] Finish training 
* [ ] Inference code 

## Reference 
https://github.com/facebookresearch/detr
