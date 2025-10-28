# MushroomML

## Description
This project aims to build a lightweight image classification system that recognizes a small set of common mushroom species from camera images in real time and deploy it on a Seeed Studio XIAO ESP32-S3 board with a connected camera. This project builds a compact image classifier for ~9 visually distinct mushroom genera and deploys it to a microcontroller with camera. The device classifies frames in real time and can fall back to an “unknown” label when not confident.

## Goals
- G1: Train a compact convolutional neural network (CNN) that classifies 9 visually distinct mushroom species with ≥85% top-1 accuracy on a held-out test set.

- G2: Quantize and deploy the model to the ESP32-S3 board, on at least 96 x 96 resolution.

- G3: Implement an “unknown” rejection option for out-of-class samples (target ≤10% false positives).

- G4: Visualize classification performance (per-class precision/recall, confusion matrix) and produce a live demo video running on the board.

## Data Collection

Using dataset:
https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images

With 9 different species with sizes:
- Lactarius      1563  
- Russula        1148  
- Boletus        1073  
- Cortinarius     836  
- Amanita         750  
- Entoloma        364  
- Agaricus        353  
- Hygrocybe       316  
- Suillus         311  

Cleaning: Normalize resolution and aspect ratio.

## Modeling Plan

### Preprocessing & feature extraction:
Resize to 128 by 128
Normalize pixel values
Data augmentation: random rotation, brightness/contrast jitter, horizontal flips, small occlusion/cutout, mild blur/noise

## Notebook Summary
- Split: Stratified 70/15/15 (train/val/test).
- Input size: IMG_SIZE = 128 (square center-crop/resize).
- Batch: BATCH = 32.
- Augmentations: Horizontal flip + brightness/contrast/saturation/hue jitter (tf.image.random_*), plus standard dtype conversion/resize.
- Balanced training: Uses tf.data.Dataset.sample_from_datasets with per-class weights to mitigate class imbalance.
- Model family: Custom MobileNet-tiny / depthwise-separable conv blocks (includes DepthwiseConv2D; MODEL_CHOICE='mbv2' in the notebook).
- Optimizer: Adam (1e-3, later 3e-4 fine-tune).
- Loss/metrics: sparse_categorical_crossentropy, accuracy.
- Quantization: Full-integer INT8 w/ representative dataset from training images; uint8 input & output; TFLite file named like:
mushroom_{testAcc:.2f}_{IMG_SIZE}_{MODEL_CHOICE}_int8.tflite.

## Current Results
- Best validation accuracy: ~66.8%.
- Test accuracy: ~69.1%

## Quantization
- Backbone: Depthwise-separable conv “tiny MobileNet” variant.
- Input: 128×128×3 (uint8 for TFLite INT8 pipeline).
- Quantization: Full-integer INT8 with a representative dataset from training images; model expects uint8 input and emits uint8 logits (softmax on-device).
- Pipeline niceties: Caching/prefetch (AUTOTUNE), class-balanced sampler.

## Deployment to XIAO ESP32-S3:
- MCU: ESP32-S3, dual-core Xtensa LX7 @ up to 240 MHz; vector instructions that ESP-NN exploits. 
ESP Component Registry

- Memory: 8 MB PSRAM + 8 MB Flash on the Sense board; microSD slot on the Sense base.

- Camera: Seeed’s Sense kit shipped historically with OV2640 (1600×1200); newer batches use OV3660 (2048×1536) due to OV2640 EOL.

Planned “Unknown” Rejection:  
Max-softmax thresholding: if max(p) < τ → label unknown.  
Optionally track an entropy threshold, and/or train with an “other” bucket from near-neighbors/out-of-set images to tighten the decision boundary.

## Test Plan

- Split: Stratified 70/15/15 (train/val/test).
- Metrics: Top-1 accuracy, macro-F1, per-class precision/recall.
- Unknown detection: Coverage–risk curve vs threshold τ.
- On-device: Verify accuracy drop after INT8 ≤5pp; measure latency/FPS and RAM usage with and without ESP-NN enabled (via perf logs).
