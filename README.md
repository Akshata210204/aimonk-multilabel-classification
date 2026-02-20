# Aimonk Multi-Label Classification Assignment

## Overview
Implemented a multi-label classification system using pretrained ResNet18 in PyTorch.

## Model
- Pretrained ResNet18 (ImageNet weights)
- Modified final layer to output 4 attributes
- Fine-tuned on provided dataset

## Handling NA Values
Images with NA attributes are not removed.
A masking mechanism is used to ignore NA labels during loss calculation.

## Handling Class Imbalance
Class imbalance handled using pos_weight in BCEWithLogitsLoss.
Weights are calculated dynamically from training dataset.

## Data Preprocessing
- Resize to 224x224
- Random Horizontal Flip
- Normalization using ImageNet mean & std

## Training
- 80/20 Train-Validation split
- Adam optimizer
- Learning rate = 0.0001
- 10 epochs
- Best model saved based on validation loss

## Outputs
- best_model.pth
- model.pth
- loss_curve.png

## Inference
Run:
python inference.py

The image path is defined inside inference.py.
You can modify the variable `image_path` to test different images.

The script prints the list of predicted attributes present for the given image.
