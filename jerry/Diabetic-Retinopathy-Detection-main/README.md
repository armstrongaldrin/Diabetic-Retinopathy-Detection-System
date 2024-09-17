. 

## Data

The dataset is obtained from a [2015 Kaggle competition](https://www.kaggle.com/c/diabetic-retinopathy-detection). The original training data consists of about 35000 images taken of different people, using different cameras, and of different sizes. The dataset is heavily biased as most of the images are classified as not having DR. Pertaining to the preprocessing section, this data is also very noisy, and requires multiple preprocessing steps to get all images to a useable format for training a model.

#Diabetic Retinopathy Detection System - Requirements Document
1. Project Overview
The Diabetic Retinopathy Detection System aims to classify medical images of retinas to detect the presence and severity of diabetic retinopathy (DR). The project uses machine learning models and image augmentation techniques to improve accuracy.

2. Objectives
Build a convolutional neural network (CNN) to classify retinal images.
Use image augmentation techniques (rotation, mirroring) to expand the dataset and enhance model performance.
Achieve a high classification accuracy on the diabetic retinopathy dataset.
3. Functional Requirements
3.1 Data Preprocessing
Input: Retinal images in JPEG format and corresponding labels.
Operations:
Load image dataset and associated labels from CSV files.
Process images by removing file extensions for easier referencing.
Split images into two categories:
No DR (level 0)
DR Present (level 1 and above)
3.2 Data Augmentation
Mirroring:
Mirror images that do not show signs of diabetic retinopathy to augment the dataset.
Mirror images with any level of diabetic retinopathy.
Rotation:
Rotate all images that show signs of diabetic retinopathy at multiple angles (90°, 120°, 180°, 270°).
File Output:
Save all augmented images in the specified directory with appropriate naming conventions (e.g., image_mir.jpeg, image_rot_90.jpeg).
3.3 Model Building
Model Structure:
Design and implement a Convolutional Neural Network (CNN) using TensorFlow/Keras.
Pre-trained models such as InceptionV3 may be leveraged for transfer learning.
Training:
Train the CNN on the augmented image dataset.
Use appropriate loss functions (e.g., categorical cross-entropy) and optimizers (e.g., Adam).
Evaluation:
Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.
3.4 User Interface
Interface for Training & Testing:
Provide an interface to initiate training on the dataset and evaluate the model.
Allow users to upload new retinal images and get predictions on the severity of diabetic retinopathy.
4. Non-Functional Requirements
4.1 Performance
The model should achieve a minimum accuracy of 90% on the test dataset.
The image processing and augmentation steps should be optimized for quick execution.
4.2 Scalability
The system should be able to handle large image datasets (up to tens of thousands of images).
4.3 Compatibility
The system should be compatible with Python 3.7+ and libraries such as TensorFlow, Keras, OpenCV, and Pandas.
The project will be developed using Google Colab or local Jupyter notebooks.
5. System Architecture
5.1 Components:
Data Layer:
Image dataset and labels stored in CSV files.
Processing Layer:
Preprocessing functions for image loading, augmentation, and splitting.
Model Layer:
CNN model for image classification.
UI Layer:
Simple command-line interface (CLI) or notebook-based interface for interaction.
6. Technical Constraints
Image size should be reduced to 256x256 pixels to reduce computational load.
Augmented images will be stored in JPEG format.
Computation-heavy tasks (e.g., training) will be performed on cloud resources (Google Colab with GPU).
7. Tools and Libraries
Programming Language: Python 3.x
Deep Learning Framework: TensorFlow/Keras
Image Processing: OpenCV, scikit-image
Data Handling: Pandas, Numpy
Hardware: Google Colab (GPU) or local machine with CUDA support
8. Timeline
Week 1: Set up project structure, gather dataset, and preprocess images.
Week 2-3: Implement image augmentation techniques (mirroring, rotation) and build initial CNN model.
Week 4: Train and evaluate the model on augmented dataset.
Week 5: Refine the model and work on optimization and documentati


