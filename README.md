# MushroomML

## Description
This project aims to build a lightweight image classification system that recognizes a small set of common mushroom species from camera images in real time and deploy it on a Seeed Studio XIAO ESP32-S3 board with a connected camera. The system will classify mushrooms into a small number of species and output an “unknown” label when uncertain.

## Goals
- G1: Train a compact convolutional neural network (CNN) that classifies at least 5 visually distinct mushroom species with ≥85% top-1 accuracy on a held-out test set.

- G2: Quantize and deploy the model to the ESP32-S3 board, on at least 240 x 240 resolution.

- G3: Implement an “unknown” rejection option for out-of-class samples (target ≤10% false positives).

- G4: Visualize classification performance (per-class precision/recall, confusion matrix) and produce a live demo video running on the board.

## Data Collection

Public mushroom image datasets such as:
https://images.cv/dataset/mushroom-image-classification-dataset
https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images

Choose 5–8 species or genera that are visually distinct (e.g., Amanita, Boletus, Coprinus, Agaricus, Pleurotus).

Cleaning: Remove duplicates, mislabeled or blurry images. Normalize resolution and aspect ratio.

## Modeling Plan

### Preprocessing & feature extraction:
Resize to 240 by 240, VGA, SVGA, etc camera supported resolutions.
Normalize pixel values
Data augmentation: random rotation, brightness/contrast jitter, horizontal flips, small occlusion/cutout, mild blur/noise

### Model:
Lightweight CNN with depthwise-separable convolutions (MobileNet-mini style)
Train in Keras/TensorFlow, use cross-entropy loss with label smoothing
Quantize to INT8 using TensorFlow Lite (QAT if needed)
I will also test out different model architectures and sizes and use the best performing one.

### Deployment:
Load quantized model as C array into ESP-IDF
Run inference with ESP-NN

## Data Exploration and Visualization Plan

### Exploratory Data Analysis:
Class distribution bar plots
Sample grid of representative images per class

### Training monitoring:
Accuracy/loss curves (train vs validation)
Confusion matrix on test set
t-SNE/UMAP projection of feature embeddings (optional)

### Deployment demo:
short video of the ESP32-S3 classifying live camera framesin nature

## Test Plan

### Dataset split:
70% training, 15% validation, 15% test, stratified by class
### Metrics:
Top-1 accuracy, macro F1 score, per-class precision/recall
Unknown detection rate (coverage vs risk curve)
### Evaluation:
Report metrics on test set after training
Verify model still meets accuracy >80% after INT8 quantization
Measure on-device inference speed (FPS) and memory usage
