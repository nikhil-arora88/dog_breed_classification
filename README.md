# dog_breed_classification

## Objective
In this project post, we will understand how CNNs can be used to classify dog breeds using Python. We will build models from scratch as well as make use of other popular models such as VGG16, ResNet50 etc. using transfer learning and compare their performances. We will make use of the popular Keras library (https://keras.io/)

This post is part of my Udacity Data Science Nanodegree capstone project. The objective is to be able to write an algorithm that takes images of dogs as inputs and predicts the breed.


## Problem Statement
As part of the capstone project, Udacity has provided us 8,351 dog images using which we need to build a classifier to predict the dog breed out of a total of 133 categories.
Additionally, Udacity has also provided us 13,233 human images. We also need to build an algorithm which determines whether the image has a human face.
The final algorithm/function should accept a file path to an image and first determine whether the image contains a human, dog, or neither. Then,
if a dog is detected in the image, return the predicted breed.
if a human is detected in the image, return the resembling dog breed.
if neither is detected in the image, provide output that indicates an error.
We shall be using Convolutional Neural Networks as they are well known for image classification tasks. We will try one model which will be built from scratch and two models using transfer learning. The transfer learning models will make use of the publicly available VGG16 (https://keras.io/api/applications/vgg/#vgg16-function) and ResNet50 (https://keras.io/api/applications/resnet/#resnet50-function) pre-trained models. For this project, Udacity has already provided the bottleneck features from these pre-trained models.

## Libraries Used/Requirements
opencv-python==3.2.0.6
h5py==2.6.0
matplotlib==2.0.0
numpy==1.12.0
scipy==0.18.1
tqdm==4.11.2
keras==2.0.2
scikit-learn==0.18.1
pillow==4.0.0
ipykernel==4.6.1
tensorflow==1.0.0

## Project Files
1. dog_app_final.ipynb: The main project python notebook with the following steps:
Step 0: Import Datasets
Step 1: Detect Humans
Step 2: Detect Dogs
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
Step 6: Write your Algorithm
Step 7: Test Your Algorithm


2. dog_app_final.pdf: PDF version of the dog_app_final.ipynb notebook

3. saved_models: Folder that has the final models saved as .hdf5 files

4. images: Folder that has some images on which the final algorithm (predict_breed function) is tested

5. requirements.txt: Text file with libraries used in the project


## Result Summary
We finalize a transfer learning model using a pre-trained ResNet50 model for creating an image classifer that predicts dog breeds (out of 133 classes). The accuracy of this model on the test data set is ~80% which is better than the VGG16 transfer learning model (test accuracy: ~45%) and the CNN model created from scratch (test accuracy: ~16%)


## Blog Post
https://niksarster.medium.com/predicting-dog-breeds-using-cnns-1f18c19de961
