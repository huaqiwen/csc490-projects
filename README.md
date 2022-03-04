# CSC490H1: Making Your Self-driving Car Perceive the World

This repository contains our implementation for Module 1 of CSC490H1:\
Making Your Self-driving Car Perceive the World.

## Overview

The detector is trained on 27 sequences of the [PandaSet](https://scale.com/open-datasets/pandaset) LiDAR dataset. \
The video below demonstrates vehicle detections on 960 frames of 12 testing sequences (different from the training sequences), with the green boxes being the ground truth labels and red boxes being the detections:

https://user-images.githubusercontent.com/37789937/156686621-7a5a93c0-ca9a-45bd-b0f3-a82aac14e6c7.mp4

## Getting started

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html):

   ```bash
   curl 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh' > Miniconda.sh
   bash Miniconda.sh
   rm Miniconda.sh
   ```

2. Close and re-open your terminal session.

3. Change directories (`cd`) to where you cloned this repository.

4. Create a new conda environment:

   ```bash
   conda env create --file environment.yml
   ```

5. Activate your new environment:

   ```bash
   conda activate csc490
   ```

6. Download [PandaSet](https://scale.com/resources/download/pandaset).
   After submitting your request to download the dataset, you will receive an
   email from Scale AI with instructions to download PandaSet in three parts.
   Download Part 1 only. After you have downloaded `pandaset_0.zip`,
   unzip the dataset as follows:

   ```bash
   unzip pandaset_0.zip -d <your_path_to_dataset>
   ```
