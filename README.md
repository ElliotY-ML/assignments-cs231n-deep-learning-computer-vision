# Background
This repo contains links to University of Michigan EECS 498.008/598.008 Deep Learning for 
Computer Vision course assignments completed by Elliot using Google Colab, Jupyter Notebook, Python, and PyTorch.

UMich EECS 498.008/598.008 builds upon the Stanford University CS231n Deep Learning for Computer Vision course,
with additional topics on emerging Computer Vision research developments. 

From the UMich course description:  
"This course is a deep dive into details of neural-network based deep learning methods for computer vision. 
During this course, students will learn to implement, train and debug their own neural networks and gain a 
detailed understanding of cutting-edge research in computer vision. We will cover learning algorithms, 
neural network architectures, and practical engineering tricks for training and fine-tuning networks for visual recognition tasks."  

[UMich EECS 498.008/598.008 Course](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/syllabus.html)  
[ Stanford CS231n Course](https://cs231n.github.io/)


# Table of Contents
1. [Assignment 1: PyTorch 101; k-Nearest Neighbor Classifier](#assignment-1-pytorch-101-knearest-neighbor-classifier)
2. [Assignment 2: Linear Classifiers; Two-Layer Neural Network](#assignment-2-linear-classifiers-two-layer-neural-network)
3. [Assignment 3: Fully-Connected Neural Network; Convolutional Neural Network](#assignment-3-fully-connected-neural-network-convolutional-neural-network)
4. [Assignment 4: Object Detection](#assignment-4-object-detection)
5. [Assignment 5: Image Captioning with Recurrent Neural Networks; Transformers](#assignment-5-image-captioning-with-recurrent-neural-networks-transformers)
6. [Assignment 6: Variational Autoencoder; Generative Adversairal Networks, Network Visualization, Style Transfer](#assignment-6-variational-autoencoder-generative-adversairal-networks-network-visualization-style-transfer)

# Assignment 1: PyTorch 101; kNearest Neighbor Classifier 
Status: Completed  Jan 2024  
[Link to GitHub Repo (Please request access to this private Repo)](https://github.com/ElliotY-ML/cs231n-assignment-1)  

Implementation of these classes and functions:  
* KnnClassifier for image data with dimensions (num_images, channels, height, width)
* k-folds cross validation 
* compute_distances using nested loops, single loop, and vectorized implementation

Key concepts and skills used:  
* PyTorch tensors manipulation
* Euclidean distance calculations

Trained K-Nearest Neighbor classifier for CIFAR-10 image dataset achieved test set accuracy: 33.86%

# Assignment 2: Linear Classifiers; Two-Layer Neural Network  
Status: Completed  Jan 2024  
[Link to GitHub Repo (Please request access to this private Repo)](https://github.com/ElliotY-ML/cs231n-assignment-2)  

Implementation of these classes and functions:
* LinearClassifier for flattened image data in tensors (num_data, flattened_dimensions)
* SVM Loss function and Softmax (Cross Entropy) Loss function forward and backward pass
* TwoLayerNet fully connected neural network for flattened image data   
* nn_forward_pass, nn_forward_backward 
* nn_train using Stochastic Gradient Descent (SGD)
* Hyperparameters search using grid search

Key concepts and skills used:   
* Devise Computational Graphs to create Forward Pass and Backward Backpropagation calculations for addition, multiplication, element selection operations
* Naive Implementations using Loops to understand element-by-element calculations 
* Vectorized Implementations to harness the speed-up of vectorized operations  
* L2 Weight Regularization  

Trained TwoLayerNet model for CIFAR-10 image dataset achieved test set accuracy: 53.79%

# Assignment 3: Fully-Connected Neural Network; Convolutional Neural Network 
Status: Completed March 2024  
[Link to GitHub Repo (Please request access to this private Repo)](https://github.com/ElliotY-ML/cs231n-assignment-3)  

Implementation of these classes and functions:  
* Linear Fully Connected; ReLU; Linear_ReLU; DropOut layers for image data in tensors
* TwoLayerNet; FullyConnectedNet with arbitrary number of hidden layers 
* SGD; SGD w/ momentum; RMSProp; Adam weight updates
* Conv; MaxPool; BatchNorm; SpatialBatchNorm layers for image data with dimensions (num_images, channels, height, width)
* ThreeLayerConvNet; DeepConvNet with arbitrary number of convolutional layers
* Kaiming weights initialization
 

Key concepts and skills used:   
* Define Autograd-like functions with forward and backward calculations
* Use Modules approach to create Fully Connected Networks and Convolutional Neural Networks with arbitrary number of layers (depth)
* Devise Computational Graphs to create Forward Pass and Backward Backpropagation calculations for Linear, ReLU, DropOut, Convolution, MaxPool, Batch Normalization layers  
* Implement KaiMing and Xavier weight initiatlizations for Fully Connected linear weights and Convolutional Filters  
* Trainable Weights update rules  

Trained DeepConvNet (10 Conv + 1 FC layer) model for CIFAR-10 image dataset achieved test set accuracy: 71.66%

# Assignment 4: Object Detection 
Status: Not Started  

Implementation of these classes and functions:  TODO  

Key concepts and skills used:  TODO  

# Assignment 5: Image Captioning with Recurrent Neural Networks; Transformers 
Status: Not Started   

Implementation of these classes and functions:  TODO  

Key concepts and skills used:  TODO  
  

# Assignment 6: Variational Autoencoder; Generative Adversairal Networks, Network Visualization, Style Transfer 
Status: Not Started  

Implementation of these classes and functions:  TODO  

Key concepts and skills used:  TODO  



