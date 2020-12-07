# Using a Variational Autoencoder for Image classification

## Introduction

The goal of this project is to develop a method of image classification using the reconstruction loss of a variational autoencoder (VAE). Although VAEs are typically used as a generative model and not for classification problems, there are some potential benefits of using a VAE for classification. If the classification problem is to distinguish just 1 class of image from many others, then for a typical approach the model would need to be given training images and labels from all the different classes. But using a VAE, the model can be trained for reconstruction using only images from the 1 class of interest. The VAE will learn the distribution of images falling into that class and if any image from another class is inputted into the model the reconstruction loss between the original image and the output image will be higher.

The specific problem addressed in this project is the classification of dog images.

## Dataset

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/dataset.jpg">
</p>

In this project the Animal Faces dataset from Kaggle is used. It contains 16,130 images of dogs, cats, and other wildlife. 4,738 images of dogs were used for training. For testing, a dataset of 500 dog images and 500 images of cats and other wildlife was used.  

## Data Preprocessing

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/PCA.jpg">
</p>

The images were resized from 512x512 to 128x128 to speed up the training process. Mean subtraction was applied to the images. A PCA model was trained on the training data to reduce the input dimensionality to 2000 principle components. Above is a comparison of an original training image and a reconstructed image generated from doing the inverse PCA transform of its principle components. The majority of the image's information is captured by the principle components.

## VAE Structure

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/VAE.jpg">
</p>

Above is the structure used for the VAE. The input is the 2000 principle components of the image. The encoding network is made up of 4 dense fully connected layers with 2048 neurons each. Next is a 512 dimensional latent space. The latent variables are sampled by the decoding layer made up of 4 dense fully connected layers with 2048 neurons. The output of the VAE are 2000 reconstructed principle components. 

## Training and Classification

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/training.jpg">
</p>

For training we preprocess and apply PCA to each image before feeding it into the VAE. The VAE learns to produce reconstructed images that are as close to the original image as possible. The mean squared error between original and reconstructed image from inverse PCA along with the Kullback–Leibler divergence are added and used as the loss function.

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/loss.jpg">
</p>

Based on the reconstruction training loss (MSE) of the model after during training, an MSE value of 200 was chosen as the threshold to determine if an image is a dog or not. Any image whos reconstructed version has an MSE below 200 is classified as a dog and other images are classified as not dog.

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/prediction.jpg">
</p>

## Results

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/recon1.jpg">
</p>

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/recon2.jpg">
</p>

Above are some examples of images from the test set and their reconstructed versions. The first example is a dog image, the VAE reconstruction looks similar to the original image although it appears blurry. It is common for VAEs to generate somewhat blurry images. The reconstructed image still resembles a dog. The second example is an image of a tiger. It is more difficult to tell that the reconstructed image is a tiger. The VAE is attempting to reconstruct a dog image based on the tiger input. Because of this the reconstruction loss is higher. 

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/results.jpg">
</p>

Above are the classification results of the model on the testing set. All images with reconstruction loss below the green threshold line were classified as dogs. The majority of the images below the threshold were ineed dogs. However, there were a good amount of non-dog images which were misclassified resulting in a false positive rate of 19.8%. This is likely because many of the other animals like foxes, cats, wolves, etc. look ver similar to a dog. The overall model accuracy was 85.5% and the false negative rate was 9.2%.

## Baseline comparison

<p align="center">
  <img src="https://github.com/JoshuaMathew/VAE-Image-Classification/blob/main/images/cnn.jpg">
</p>

A convolutional neural network was trained on the same dataset to act as a baseline for comparison. The CNN achieved an overall classifcation accuracy of 90%.

## Conclusion

This project demonstrates that a VAE can be utilized as comparable alternative for image classification when compared to other mainstream methods such as a CNN. In certain cases VAEs also have the added benefit of only requiring images from 1 class for training.
