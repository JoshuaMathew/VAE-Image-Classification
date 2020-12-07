# Using a Variational Autoencoder for Image classification

## Introduction

The goal of this project is to develop a method of image classification using the reconstruction loss of a variational autoencoder (VAE). Although VAEs are typically used as a generative model and not for classification problems, there are some potential benefits of using a VAE for classification. If the classification problem is to distinguish just 1 class of image from many others, then for a typical approach the model would need to be given training images and labels from all the different classes. But using a VAE, the model can be trained for reconstruction using only images from the 1 class of interest. The VAE will learn the distribution of images falling into that class and if any image from another class is inputted into the model the reconstruction loss between the original image and the output image will be higher.

The specific problem addressed in this project is the classification of dog images.

## Dataset

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/dataset.jpg">
</p>

In this project the Animal Faces dataset from Kaggle is used. It contains 16,130 images of dogs, cats, and other wildlife. 4,738 images of dogs were used for training. For validation, a dataset of 500 dog images and 500 images of cats and other wildlife was used.  

## Data Preprocessing

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/PCA.jpg">
</p>

The images were resized from 512x512 to 128x128 to speed up the training process. Mean subtraction was applied to the images. A PCA model was trained on the training data to reduce the input dimensionality to 2000 principle components. Above is a comparison of an original training image and a reconstructed image generated from doing the inverse PCA transform of its principle components. The majority of the image's information is captured by the principle components.
