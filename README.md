# Neural Network for Diabetes Prediction

<img src="https://github.com/vinitshetty16/Neural-Network-for-Diabetes-Prediction/assets/63487624/30a7f070-42c4-4cd0-8607-3ee2089fa432" alt="Neural Network" style="width:1400px; height:400px;">

## Overview

This repository contains code and resources for constructing a forward neural network for the task of image classification, specifically focused on diabetes prediction. The neural network model is trained on the diabetes prediction dataset using PyTorch and TensorFlow frameworks.

## Methodology

The methodology involves the following steps:

1. **Importing Libraries and Packages:** Importing necessary libraries and packages for data manipulation, visualization, model construction, and evaluation.

2. **Data Exploration:** Describing the target classes for the prediction task and displaying examples from each target class to identify patterns.

3. **Data Preprocessing:** Preprocessing the dataset by handling missing values, one-hot encoding categorical variables, and standardizing numeric features.

4. **Preparing Data for Learning:** Splitting the dataset into training, validation, and test sets and preparing them for neural network training.

5. **Model Construction:** Designing a deep feedforward neural network using TensorFlow and Keras. Configuring the model architecture, including input, hidden, and output layers, as well as activation functions.

6. **Model Fitting:** Compiling the model with appropriate loss function, optimizer, and metrics. Training the model on the training data and monitoring the training process.

7. **Model Evaluation:** Evaluating the trained model on the validation and test sets to assess its performance in terms of loss and accuracy.

## Dependencies

- PyTorch
- Python 3.x
- umap-learn
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## TensorBoard Analysis

The TensorBoard analysis evaluates the training process of the neural network model using TensorBoard. Overfitting is observed around epoch 6 and 7, indicating a gap between training and validation performance.

## Regularization Techniques

Two regularization techniques, batch normalization and dropout, are implemented to improve model performance:

- **Batch Normalization:** A batch normalization layer is added after each hidden layer. This helps control overfitting and enhances generalization.
- **Dropout:** Dropout layers are introduced after each hidden layer to randomly drop connections during training, preventing overfitting.

## Summary and Insights

- Regularization techniques effectively mitigate overfitting and improve model generalization.
- Deeper layers capture more intricate features, enhancing class separation and clustering in the embedding space.
- Embeddings in the final layer show decreased discriminative power, indicating the need for better feature capture in intermediate layers.

## Conclusion

The construction of a forward neural network for diabetes prediction demonstrates the application of deep learning techniques in healthcare. By leveraging neural networks, we can effectively analyze medical data and make predictions about disease outcomes. The use of both PyTorch and TensorFlow frameworks provides flexibility and scalability in model development and deployment.

## Usage

To replicate the analysis:

1. Clone this repository.
2. Install the necessary dependencies.
3. Run the provided Python scripts for training, regularization, and analysis.
4. View the results and insights in the Jupyter Notebook or Markdown report.
