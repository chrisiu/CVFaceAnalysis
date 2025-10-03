# Fake Face Detection with Various CNN Architectures 
## Chris Haleas and Samuel Salama

This repository contains the implementation of several Convolutional Neural Network (CNN) architectures to detect fake/real face images. Each model is tested for its robustness against different adversarial attacks.

## Project Overview

Each notebook in our repository follows the same workflow:

1. Data loading and preprocessing
2. Model setup (using a specific pre-trained architecture)
3. Model fine-tuning
4. Performance evaluation on clean validation data
5. Adversarial attack implementation (Gaussian noise, color shift, black patch)
6. Evaluation of model robustness against each attack type

## Model Implementations

The following architectures have been implemented:

| File                  | Architecture           |
| --------------------- | ---------------------- |
| densenet121.ipynb     | DenseNet121            |
| efficientNet_b0.ipynb | EfficientNet B0        |
| inceptionv1.ipynb     | GoogLeNet/Inception v1 |
| regnet_x_400mf.ipynb  | RegNet X 400MF         |
| resnet-18.ipynb       | ResNet-18              |
| squeezenet1_1.ipynb   | SqueezeNet 1.1         |

## Dataset

This project uses the Real and Fake Face Detection dataset. The dataset contains real face images and fake face images of different difficulty levels (easy, mid, hard).

Note: The "real_and_fake_face_example" folder contains sample data from the dataset, located in the "training_fake_example" and "training_real_example" subfolders. This folder is designed to replicate the structure of the Kaggle dataset and provide example images. It is not a substitute for downloading the full dataset.

### Dataset Acquisition

You can download the dataset from Kaggle: [Real and Fake Face Detection Dataset](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)

## Usage Instructions

1. Clone this repository to your local machine
2. Download the dataset from Kaggle (the dataset )
3. Update the file paths in each notebook:
    
    ```python
    # Change these paths to your own directoriesreal_dir = "/path/to/your/training_real"fake_dir = "/path/to/your/training_fake"
    ```
    
4. Run the notebooks to train and evaluate each model

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- numpy
- PIL (Pillow)

## Device Support

The code automatically selects the appropriate device (MPS for Apple Silicon, or CPU if MPS is unavailable):

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

For NVIDIA GPUs, you can modify this to:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Research Paper 
You can view the PDF of our research paper, 'FinalProject_B457.pdf,' in the repository.