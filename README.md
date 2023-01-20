# Pytorch-Landscape-Segmentation

Welcome to this repository, which contains the solution to a remote sensing problem using deep learning techniques. The main objective of this challenge is to segment images acquired by a small UAV (sUAV) in the area of Houston, Texas. These images were acquired in order to assess the damages on residential and public properties after Hurricane Harvey. In total, there are 25 categories of segments (e.g. roof, trees, pools, etc.).

This repository contains the following files:

    resizing-dataset.py: This file contains the code for data augmentation techniques used to increase the size of the dataset.

    unet_model.py: This file contains the code for the model architecture used for image segmentation.

    pipeline-data-augmentation.py: This file contains the code for the pipeline used to create and train the model that uses data augmentation.

    pipeline-submitted-model.py: This file contains the code for the pipeline used to create and train the updated model
    
    Segment features around residential buildings in UAV images.pdf: The final report of the project

To use this repository, please make sure you have the required dependencies installed. The code is written in Python and uses PyTorch library. You should be able to run the code by running the pipeline file after downloading the dataset. The dataset can be downloaded [MSC AI 2022 Competition](https://www.kaggle.com/competitions/msc-ai-2022/overview).
