# AI-Generated Image Detection and Robustness Analysis

A computer vision project using ResNet50 transfer learning in PyTorch to classify real and AI-generated images, followed by an investigation of the detector’s sensitivity to optimized image modifications.

Developed for CS 675: Computer Vision at the University of Massachusetts Boston.

## Overview

This project explores two questions:

1. How accurately can a pretrained ResNet50 distinguish real images from AI-generated images on CIFAKE?
2. Can constrained image modifications change the detector’s predictions?

The project combines binary image classification with an optimization-based robustness experiment.

## Dataset

The project uses **CIFAKE**, containing:

- **100,000 training images**
- **20,000 test images**
- Two classes: **REAL** and **FAKE**

Images are resized to 224 × 224 pixels and normalized using ImageNet mean and standard deviation values.

The dataset is not included in this repository.

## Approach

### Image Classification

- Used a pretrained ResNet50 model with most pretrained layers frozen.
- Replaced the final fully connected layer with a two-class classification head.
- Trained using the Adam optimizer and cross-entropy loss.
- Evaluated predictions using accuracy, a confusion matrix, and a classification report.

### Robustness Analysis

Selected AI-generated images were iteratively modified while keeping the detector’s parameters fixed.

The optimization combined:

- A classification loss encouraging predictions toward the REAL class.
- A regularization term penalizing changes to the original image.

This experiment examined whether constrained modifications could reduce the detector’s confidence in the FAKE class.

## Results

Results reported in the project report:

| Metric | Accuracy |
|--------|----------|
| Training accuracy | 97.76% |
| Test accuracy | 97.20% |

Experiments were conducted using an NVIDIA GeForce RTX 4050 Laptop GPU.

For selected examples, optimized image modifications shifted predictions toward the REAL class. These observations illustrate a robustness limitation; they do not demonstrate improved robustness or establish an aggregate attack success rate.

## Technologies

- Python
- PyTorch
- ResNet50
- Transfer learning
- Image preprocessing
- Optimization-based robustness evaluation

## Limitations

- Evaluation was conducted on CIFAKE; performance on other datasets or unseen image generators has not been established.
- Resizing low-resolution images to 224 × 224 pixels does not recover additional image detail.
- The robustness experiment provides observations on selected examples rather than a comprehensive benchmark.
- High classification accuracy on the original test set does not imply reliable detection of modified images.
- The project is a research prototype, not a production-ready image authenticity service.

## Reproducibility

Exact environment requirements, dataset setup instructions, and commands for training, evaluation, and robustness analysis will be added after the repository’s scripts and dependencies are verified.

## Report

The accompanying report, **AI-Generated Image Detection with Post-Processing Robustness Analysis**, describes the methodology, experimental results, and limitations.

## Author

**Vrishin Jain**  
MS in Computer Science  
University of Massachusetts Boston

## Acknowledgments

This project uses the CIFAKE dataset and a pretrained ResNet50 model. Related research and references are documented in the project report.
