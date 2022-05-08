# Sea state estimation from videos 
## Overview
Repo for a project at Uni Bremen about estimating sea state parameters from videos (Project "Seegangsmessgeraet" at Uni Bremen).

## CNN for image classification
### 1. Collecting and preparing data
#### downloading.py
Script for downloading image files of pubished data set at https://data-dataref.ifremer.fr/stereo/ (Guimarães, P. V. et al. (2020) ‘A data set of sea surface stereo images to resolve space-time wave fields’, Scientific Data, 7(1), pp. 1–12. doi: 10.1038/s41597-020-0492-9)
Either put baseline website address if image names follow sythax [upcounting number]_01.tif for cam 1 or get http-Textfile first and use this for parsing and browse this for getting the images url for downloading.

#### converting.py
Converts images in given directory into desired format, here converting .tif or .png into .jpeg, saving the new images in new directory.

#### cropping.py
Crops images in given directory and saves it in new directory. Cutting out border area for not having camera pieces / horizon in the image. Applies the same cropping to all images in directory.

### 2. Setting up, training and testing CNN-classifier

Following code is supposed to be executed by an interpreter running on gpu as it requires a lot of resources.
#### training.py
Establishing and training an CNN classifier with vgg16-architecture from scratch.
includes up-to-date prediction code with evaluation metrics overall accuracy, precision, recall and building confusion matrices with diagramms respectively.

#### outdated: prediction.py
Loading a given model for making predictions on input data. Outdated, as version incompatibilies made loading a previously saved model impossible.
