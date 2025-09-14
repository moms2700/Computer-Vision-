# Computer Vision - Deep Learning Projects

This repository contains my Deep Learning practical work, focusing on neural networks, convolutional neural networks (CNNs), and Vision Transformers.

## 📚 Contents

### 1. Introduction to Neural Networks (Lab 1-a and 1-b)
- **Authors**: William Khieu & Akli Mohamed Ait-Oumeziane
- **Date**: October 2024

#### Theoretical Part
- Neural network architecture (Forward/Backward)
- Loss functions (Cross-entropy, MSE)
- Optimization algorithms (SGD and variants)
- Backpropagation

#### Implementation
- Building a neural network from scratch
- Manual forward and backward passes
- Mini-batch SGD optimization
- Experiments with different hyperparameters

### 2. Convolutional Neural Networks (Lab 1-c and 1-d)
- **Dataset**: CIFAR-10
- **Architecture**: CNN with 3 convolutional layers + 2 fully-connected layers

#### Implemented improvement techniques:
- ✅ Data normalization
- ✅ Data Augmentation (RandomCrop, HorizontalFlip, ColorJitter, etc.)
- ✅ Learning Rate Scheduling (ExponentialLR, CosineAnnealing)
- ✅ Dropout regularization
- ✅ Batch Normalization
- ✅ Optimizer comparison (SGD, Adam, SGD+Momentum)
Course 1 (Sept. 24, 2025)

Introduction to Computer Vision and ML basics slides1
Visual (local) feature detection and description
Bag of Word Image representation
Linear classification (SVM) slides2_SVM
Course 2
Introduction to Neural Networks (NNs) slides2_NN
NN training and Statistical decision theory slides3_NN

Course 3

Datasets, benchmarks and evaluation slides3_DATA
Neural Nets for Image Classification slides4_LargeConvNet

Course 4

Large Neural Nets for Image Classification slides5_LargeConvNet

Complements for intialization of NNs and normalization: BatchNorm, LayerNorm (convnet or transformer), InstanceNorm

Course 5

Vision Transformers slides_vit1
Details of Vision Transformers architecture: Embedding matrix, positional encoding, attention module, FFN module, Norms, classToken, …

Course 6
Segmentation with CNN slides_segmenta
Transfer learning and Domain adaptation slides7_Transfer

Course 7
Vision-Language models slides10.pdf

Course 8
Explaining&Monitoring VLMs

Course 9
Self-Supervised Learning in Vision

Course 10
Generative models with GANs slides8.pdf

Course 11 (Jan. 07)
Control 2pm-3:45pm + practicals 4pm-6pm

Conditional GANs and Teaser Diffusion models slides9.pdf

Course 12 Diffusion models for Image Generation (Alasdair)
Course 13 Bayesian deep learning (Clement)
Course 14 Failure and ood detection (Clement)
#### Results
- **Baseline accuracy**: ~72.82%
- **Final accuracy**: ~83.88%
- Significant reduction in overfitting

### 3. Vision Transformers (Lab 1-e)
- **Architecture**: Implementation of the original Vision Transformer (ViT)
- **Reference**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

#### Implementation features:
- Simplified, smaller version than the original paper
- From-scratch implementation of attention mechanism
- Patch embedding and positional encoding
- Multi-head self-attention
- Adapted from the [Timm library](https://github.com/rwightman/pytorch-image-models)

#### Notes:
- Naive implementation without complex data augmentation
- Without advanced regularizations used in [DeiT and CaiT](https://github.com/facebookresearch/deit)
- Focus on understanding the architecture rather than performance

## 🛠️ Technologies Used
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebooks

## 📊 Project Structure
Computer-Vision/
├── 1_ab_Intro_to_NNs.ipynb       # Introduction to neural networks
├── 1_cd_CNNs.ipynb                # Convolutional networks
├── 1_e_Transformers.ipynb         # Vision Transformers (ViT)
├── TP.ipynb                       # Additional practical work
├── Untitled*.ipynb                # Test and experimentation notebooks
└── Reports/                       # Detailed PDF reports
├── TP1_ab_Report.pdf
└── TP1_cd_Report.pdf

## 🚀 Installation and Usage

1. Clone the repository:
```bash
git clone git@github.com:moms2700/Computer-Vision-.git
cd Computer-Vision-

Install dependencies:

bashpip install torch torchvision numpy matplotlib jupyter

Launch Jupyter Notebook:

bashjupyter notebook
📈 Performance and Experiments
Classic Neural Networks

Analysis of hyperparameter impact (batch size, learning rate)
Comparison between different activation functions

CNNs

Loss and accuracy evolution with different regularization techniques
Impact of data augmentation on generalization

Vision Transformers

Understanding the attention mechanism
Analysis of positional encoding importance
Comparison with convolutional approaches

🔍 Key Learning Points

Architectural progression: From multilayer perceptron to Transformers
Importance of regularization: Dropout, Batch Norm, Data Augmentation
Optimization: Impact of schedulers and different optimizers
Vision Transformers: New approach without convolutions for vision

👥 Contributors

Akli Mohamed Ait-Oumeziane
William Khieu

📝 License
This project is completed as part of academic coursework.
🔗 References

Vision Transformer (ViT)
DeiT: Data-efficient Image Transformers
Timm Library
