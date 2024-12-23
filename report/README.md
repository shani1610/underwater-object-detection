# Underwater Report 

## Shani Israelov 
 
we use a pretrained model:
# Load Faster R-CNN model
```
fasterrcnn = fasterrcnn_resnet50_fpn(pretrained=True)
```

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

torchinfo to print the model summary

training:
# Optimizer and Learning Rate
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# results from training 

running on the training data we have 
Epoch 20/20, Loss: 165.5529
this is high training loss. 

when running on the val dataloader we get 
Precision: 0.3974
Recall: 2.9841
mAP: 1.2619

some images:


we have several problems:
1) training data is small
2) precision is low
3) loss is high

what can we do? 
1) train more layers (which?)
2) add augmentations
3) cross validation for fine tuning

