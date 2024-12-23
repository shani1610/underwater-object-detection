# Underwater Report 

## Shani Israelov 

This is self study case study i did to practice object detection metodology. i used the underwater dataset from kaggle, 
it is a small dataset representing challenges we can face in a lot of real case scenarios. 
im using faster rcnn model, at the first step i only changed the last layer to fit the 8 categories and then finetune it. 

we use a pretrained model:

## Faster R-CNN model

there are 4 faster rcnn model builders in [PyTorch](https://pytorch.org/vision/master/models/faster_rcnn.html):
1) fasterrcnn_resnet50_fpn(*[, weights, ...]) Faster R-CNN model with a ResNet-50-FPN backbone from the Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks paper.
2) fasterrcnn_resnet50_fpn_v2(*[, weights, ...]) Constructs an improved Faster R-CNN model with a ResNet-50-FPN backbone from Benchmarking Detection Transfer Learning with Vision Transformers paper.
3) fasterrcnn_mobilenet_v3_large_fpn(*[, ...]) Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone.
4) fasterrcnn_mobilenet_v3_large_320_fpn(*[, ...]) Low resolution Faster R-CNN model with a MobileNetV3-Large backbone tuned for mobile use cases.

first im using the first option:
```
fasterrcnn = fasterrcnn_resnet50_fpn(pretrained=True)
```

add explanation of faster rcnn model. 

- Backbone with FPN
    - Intermediate Layer Getter - is the body, here is the ResNet50
    - Feature Pyramid Network
- Region Proposal Network
    - RPN Head    
    - Anchor Generator
- ROI Heads
    - Multi Scale
    - Two MLP
    - Box Predictor

we modify the box predictor to adjust the number of classes
```
num_classes = 8  # 7 classes + 1 background
in_features = fasterrcnn.roi_heads.box_predictor.cls_score.in_features  # Get the number of input features
fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # Replace with new predictor
```

```
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze specific layers
for param in model.model.roi_heads.box_predictor.parameters():  # Access the underlying model
    param.requires_grad = True
```

torchinfo to print the model summary:

<img src="./assets/run1_trainable.png" alt="drawing" width="700"/>

## training:
```
# Optimizer and Learning Rate
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

```

## results
running on the training data we have 
Epoch 20/20, Loss: 165.5529
this is high training loss. 

when running on the val dataloader we get 
Precision: 0.3974
Recall: 2.9841
mAP: 1.2619

some images:

<img src="./assets/run1_pred1.png" alt="drawing" width="200"/>
<img src="./assets/run1_gt1.png" alt="drawing" width="200"/>

<img src="./assets/run1_pred2.png" alt="drawing" width="200"/>
<img src="./assets/run1_gt2.png" alt="drawing" width="200"/>

we have several problems:
1) training data is small
2) precision is low
3) loss is high

what can we do? 
1) train more layers (which?)
2) add augmentations
3) cross validation for hyper parameter tuning
4) try other optimizer, learning rate, epoch number

