# Traffic Sign Recognition Project

This project focuses on classifying traffic signs using a Convolutional Neural Network (CNN). The dataset consists of images and their corresponding annotations.

### Instructions

#### 1. Setup
Before you start the training, make sure you have all required libraries installed and is using Kaggle environment. The key libraries used in this project are:
- numpy
- pandas
- cv2 (OpenCV)
- tensorflow (Keras)
- matplotlib
- seaborn

To install any missing libraries, run:


#### 2. Training
Ensure you have the train and test datasets ready in the respective directories. For example:
- train_image_folder_path: The directory containing training images.
- test_image_folder_path: The directory containing test images.

The CNN model for traffic sign recognition is implemented with the following architecture:
- Conv2D layers for feature extraction
- MaxPooling2D for downsampling
- Flatten layer to convert the 2D feature maps to 1D
- Dense layers for classification

To start training:
1. Load the preprocessed training data and split it into training and validation sets.
2. Compile the model and define the optimizer (e.g., Adam), loss function (e.g., categorical_crossentropy), and evaluation metric (accuracy).
3. Fit the model with the training data and validate on the validation set.

#### 3. Evaluation
After training, evaluate the model on the test set:
- Compute metrics like accuracy, precision, recall, and F1-score.
- Use a confusion matrix to analyze classification performance visually.

Sample code for evaluation:

#### 4. Hyperparameter Tuning
We use **Keras Tuner** to find the best hyperparameters for the model. To install the tuner, run:

This will allow you to experiment with different combinations of layers, units, activation functions, etc.

#### 5. Model Checkpoints
During training, you can save the best model using:

This ensures that the best model based on validation performance is saved for later use.

#### 6. Visualization
Use matplotlib and seaborn to plot confusion matrices and visualize training history (accuracy and loss curves). Example: