# Convolutional Autoencoder to reduce Noise in Computed Tomography of patients with Breast Cancer

This repository contains a neural network based on a convolutional autoencoder that allows to eliminate Gaussian and Poisson noise in computed tomography images.

## Code

This script is designed to run in Google Colab and use the GPU to run it.
* File: Autoencoder_breast_2_0_.ipynb
* To load the set of CT images, the directory where they are located must be indicated in the ld.dataset function
* The load_dataset.py code allows you to generate the dataset with these images.

## Results

* The results generated by the network show how it allows to eliminate Gaussian and Poisson noise in the test images. However, the network must be trained for a greater number of epochs, in order to improve the details and errors of the generated image. 
* The limitation of this is in the execution time that Google Colab allows with the use of GPU, which is only a couple of hours. 
* In the images folder, you can see some results.
* The loss plot shows how the training and evaluation with the test images converge towards a decrease in the error of the network predictions.
