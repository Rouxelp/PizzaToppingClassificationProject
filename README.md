# Project: Pizza Classification and Ingredient Recognition

## Context
This project utilizes Kaggle datasets and Python notebooks running on GPUs to train two distinct models:
1. An image classification model to detect the presence of a pizza.
2. A model to identify the ingredients of a pizza.
3. [BONUS]: Connect to the model outputs a price calculator and a carbon calculator (the latter in a private repository) to detect, from a single image, whether it's a pizza, list the ingredients and their quantities, and then predict the pizza's cost and carbon impact.

The final objective is to integrate these two models to build an application capable of retrieving a pizza image, identifying if it’s a pizza, and listing the main ingredients.

---

## Project Steps

### 1. Retrieving the "Pizza or Not Pizza" Dataset
- **Objective**: Download the "Pizza or Not Pizza" dataset from Kaggle for training the first model.
- **Steps**:
  - Import the dataset via the Kaggle API.
  - Prepare images for training (preprocessing, resizing).

### 2. Training and Optimization of the "Pizza or Not Pizza" Model
- **Objective**: Train a classification model to detect whether an image contains a pizza.
- **Steps**:
  - Define the model architecture (e.g., CNN).
  - Use GPUs to accelerate training.
  - Optimize hyperparameters (e.g., learning rate, number of epochs).

### 3. Model Evaluation and Visualization Tool
- **Objective**: Evaluate the model's performance and visualize some example predictions.
- **Steps**:
  - Calculate performance metrics (precision, recall, F1-score).
  - Use a notebook to visualize examples of correct and incorrect predictions.

### 4. Retrieving the "Pizza Ingredients" Dataset
- **Objective**: Download the pizza ingredients dataset for training the second model.
- **Steps**:
  - Import the dataset via Kaggle.
  - Process the data to make it compatible with the ingredient identification model.
  

### 5. Training and Optimization of the "Pizza Ingredients" Model
- **Objective**: Train a model to recognize ingredients in pizza images.
- **Steps**:
  - Design the model for ingredient identification. In step 4, we now have imported the dataset with labels, and we then have to build the technology to put the localization boxes on these images. In that way, we have the possibilty to measure the quality of the output of the model during training, by minimizing the distance between both reality and prediction, and the value of the label.
  - Optimize training using GPUs on Kaggle.
  - Adjust hyperparameters to improve prediction accuracy.

### 6. Test and Visualization of Ingredient Predictions
- **Objective**: Test the ingredient model and visualize the predictions.
- **Steps**:
  - **Evaluation**: Calculate performance metrics such as precision, recall, and F1-score.
  - **Prediction Visualization**: Implement an algorithm to take an image as input and return it with labeled or segmented ingredients.
    - **Algorithm**:
      1. Load the trained ingredient model.
      2. Preprocess the input image (resize, normalize, etc.).
      3. Apply the model to predict ingredients.
      4. Create a segmentation map based on predicted ingredients.
      5. Overlay the segmentation map on the original image.
      6. Return the labeled image.
  - **Example Visualization**: Display an example image with ingredient labels.

### 7. Integration of Both Models
- **Objective**: Integrate the "Pizza or Not Pizza" and "Pizza Ingredients" models for a comprehensive classification.
- **Steps**:
  - Create a function that first applies the pizza detection model, followed by the ingredient model.
  - Handle cases where the image does not contain a pizza (detected as "not pizza").

### 8. Building a Complete Example
- **Objective**: Set up an end-to-end workflow for pizza and ingredient recognition.
- **Steps**:
  - Create a function to retrieve a pizza image (from a URL or by upload).
  - Pass the image through the model pipeline to detect pizza and list ingredients.
  - Present the final result in a notebook, with visualizations of the intermediate steps.

---

## Technologies and Tools
- **Language**: Python
- **Frameworks**: TensorFlow or PyTorch
- **Execution Environment**: Kaggle Notebooks with GPU support
- **Visualization**: Matplotlib, Seaborn for results display
- **Dataset Access**: Kaggle API

---

## Deliverables
- **Kaggle Notebooks**: Containing each step of the process.
- **Trained Models**: Exported model files for pizza detection and ingredient identification.
- **Documentation**: Explanation of each step and results.

---

This project will provide hands-on experience in training image classification models and integrating them into a complete image analysis pipeline.
