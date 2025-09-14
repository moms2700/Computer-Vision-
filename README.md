Computer Vision - Deep Learning Projects
This repository contains my Deep Learning practical work, focusing on neural networks, convolutional neural networks (CNNs), and Vision Transformers.
📚 Contents
1. Introduction to Neural Networks (Lab 1-a and 1-b)

Authors: William Khieu & Akli Mohamed Ait-Oumeziane
Date: October 2024

Theoretical Part

Neural network architecture (Forward/Backward)
Loss functions (Cross-entropy, MSE)
Optimization algorithms (SGD and variants)
Backpropagation

Implementation

Building a neural network from scratch
Manual forward and backward passes
Mini-batch SGD optimization
Experiments with different hyperparameters

2. Convolutional Neural Networks (Lab 1-c and 1-d)

Dataset: CIFAR-10
Architecture: CNN with 3 convolutional layers + 2 fully-connected layers

Implemented improvement techniques:

✅ Data normalization
✅ Data Augmentation (RandomCrop, HorizontalFlip, ColorJitter, etc.)
✅ Learning Rate Scheduling (ExponentialLR, CosineAnnealing)
✅ Dropout regularization
✅ Batch Normalization
✅ Optimizer comparison (SGD, Adam, SGD+Momentum)

Results

Baseline accuracy: ~72.82%
Final accuracy: ~83.88%
Significant reduction in overfitting

3. Vision Transformers (Lab 1-e)

Architecture: Implementation of the original Vision Transformer (ViT)
Reference: An Image is Worth 16x16 Words

Implementation features:

Simplified, smaller version than the original paper
From-scratch implementation of attention mechanism
Patch embedding and positional encoding
Multi-head self-attention
Adapted from the Timm library

Notes:

Naive implementation without complex data augmentation
Without advanced regularizations used in DeiT and CaiT
Focus on understanding the architecture rather than performance

🛠️ Technologies Used

Python 3.x
PyTorch
NumPy
Matplotlib
Jupyter Notebooks

📊 Project Structure
Computer-Vision/
├── 1_ab_Intro_to_NNs.ipynb       # Introduction to neural networks
├── 1_cd_CNNs.ipynb                # Convolutional networks
├── 1_e_Transformers.ipynb         # Vision Transformers (ViT)
├── TP.ipynb                       # Additional practical work
├── Untitled*.ipynb                # Test and experimentation notebooks
└── Reports/                       # Detailed PDF reports
    ├── TP1_ab_Report.pdf
    └── TP1_cd_Report.pdf
🚀 Installation and Usage

Clone the repository:

bashgit clone git@github.com:moms2700/Computer-Vision-.git
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
