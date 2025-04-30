# Vehicle Damage Detection Classification

This repository contains a deep learning model for classifying vehicle damage into three categories: **minor**, **moderate**, and **severe**. The model is trained using the **InceptionV3** architecture, which is pre-trained on ImageNet, and fine-tuned for vehicle damage classification. The goal is to predict the severity of vehicle damage based on an input image and later it was updated to provide an estimated repair cost for each category.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Steps to Run](#steps-to-run)
- [Installation](#installation)


## Project Description

This project uses a convolutional neural network (CNN) built on top of the **InceptionV3** model, leveraging transfer learning. The model is trained on a dataset of vehicle damage images, which is divided into **training**, **validation**, and **test** sets.

- **Minor Damage**: Small scratches, dents, or minor cracks.
- **Moderate Damage**: Larger dents, moderate scratches, or cracks.
- **Severe Damage**: Extensive damage, large cracks, or broken parts.

![WhatsApp Image 2025-04-29 at 11 52 44 AM](https://github.com/user-attachments/assets/b914204f-fa34-44c1-82ec-7670297dc254)
![WhatsApp Image 2025-04-29 at 11 52 45 AM](https://github.com/user-attachments/assets/95132e40-fa4d-4145-bc08-b8cefb08c07a)
  

## Dataset

The dataset used in this project is located in the `data3a` directory, which contains the following folders:
- `training`: Training images for model training.
- `validation`: Validation images for model validation.
- `test`: Test images to evaluate the final model.

The images are labeled into three categories: `01-minor`, `02-moderate`, and `03-severe`.

## Steps to Run

Run the following commands on **Google Colab**, ensuring a fast internet speed:


1. from google.colab import drive
drive.mount('/content/drive')

2. !pip install keras

3. May or may not opt for updates.

4. ! python /content/drive/path/to/main.py

5. ! python /content/drive//path/to/run.py

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ayeshaakram88/Vehicle-Damage-Detection-Classification.git
   cd Vehicle-Damage-Detection-Classification
