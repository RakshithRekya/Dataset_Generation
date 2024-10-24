# Dataset_Generation

# Dataset Generation with Custom Face Attributes 

# Here’s a professional summary of my project "Dataset Generation":

# This project focuses on generating a custom dataset comprising facial images of six individuals, each annotated with five key attributes: black hair color present, mustache, gender, presence of glasses, and presence of a beard. 

# The dataset includes 2,559 images captured with varied orientations, lighting conditions, and poses, incorporating data augmentation techniques such as Gaussian blur and horizontal flipping to enhance model generalization.

# The project utilizes the PyTorch framework to implement two distinct approaches:

# Class Prediction: Using Convolutional Neural Networks (CNN) and Linear Neural Networks (LNN), this approach predicts the class (person) of each image.

# Attribute Prediction: In this approach, CNN and LNN models are employed to predict the attributes associated with the images, focusing on binary classification for features like gender, glasses, and hair color.

# The performance of both models was evaluated using metrics such as accuracy vs precision, confusion matrices, and softmax plots. 

# The CNN model outperformed the LNN model in both approaches, achieving higher accuracy and better generalization, as seen in the plotted results.
