# 🧠 MNIST Digit Recognition Suite

## Project Overview
This repository contains two interactive Deep Learning applications built using Streamlit and a CNN Keras model (Sequential). This project marks the successful completion of my first major milestone in the Deep Learning journey, focusing on the classic MNIST Dataset for handwritten digit recognition.
The primary goal was to train a high-accuracy model and deploy it in two real-world scenarios to test its robustness against variations in human handwriting.

## Applications Included

### 1. MNIST Digit Detector
This application allows users to interactively test the trained CNN model by drawing a single digit.

#### Functionality:
- Input: Users draw a single digit (0-9) on a drawing canvas.
- Model: A Keras Sequential CNN model predicts the digit instantly.

#### Purpose: Provides a visual and engaging way to verify the model's performance on live, never-before-seen input.

### 2. Digit Arithmetic Calculator
This application takes the digit recognition concept further by integrating arithmetic operations with the same CNN Keras model.

#### Functionality:
- Input: Two separate drawing canvases allow users to input two distinct single digits.
- Prediction: The model predicts both digits.  
- Visualization: A bar chart displays the predicted probability values for each digit.
- Arithmetic: Users select an operation (e.g., addition, subtraction), and the application calculates and displays the final result of the operation on the predicted numbers.

#### Challenge: This tested the ability to integrate two model inference steps into a single, cohesive application flow.

## Technical Details
- Model Architecture: Convolutional Neural Network (CNN) - Keras Sequential Model.
- Model Accuracy: Achieved a high testing accuracy of ~98-99%.

### Frameworks:
- Frontend/Deployment: Streamlit (for interactive web interface).
- Deep Learning Library: Keras / TensorFlow.
- Language: Python.
