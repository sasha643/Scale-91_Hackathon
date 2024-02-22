# AI - Enhanced Threat Detection

## Introduction

This repository implements a deepfake detection system using face recognition techniques and neural networks. The system processes videos, extracts frames, detects faces, and employs a pre-trained neural network for classification.

## Prerequisites

Make sure to install the required libraries and packages before running the code:

```bash
pip install opencv-python numpy tensorflow face_recognition torch torchvision tqdm matplotlib seaborn
```

## Usage

- Clone the repository:

```bash
git clone https://github.com/sasha643/Scale-91_Hackathon.git
```
## Components

- Object Detection Setup: Load the pre-trained object detection model.
- Frame Extraction: Extract frames from video files.
- Face Detection and Cropping: Use face recognition to detect faces in frames and crop them.
- Data Preprocessing: Preprocess the frames for feeding into the neural network.
- Neural Network Model: Load and initialize the neural network model for classification.
- Training: Train the neural network on the provided dataset.
- Testing: Evaluate the model on a separate test dataset.
- Visualization: Visualize the detection results and generate a confusion matrix.

## SDN list

Our approach assigns pseudo numbers to accounts, allowing AI to identify and monitor risks without accessing restricted PII like account holder names. This surpasses KYC, providing a pioneering solution for fintech security challenges, even in high-profile cases like "Dawood".

### Developed by: 

- Saurabh Sharma
- Dhruvrajsinh Solanki
