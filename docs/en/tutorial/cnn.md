# Tutorial: Training a CNN for Item Classification

This tutorial explains how to train a simple Convolutional Neural Network (CNN) to classify items within the CustomGrid environment. This is a great starting point for students to learn about computer vision and how it can be integrated into larger software systems.

## Overview

In our environment, the agent can encounter "dogs" and "flowers". While the environment knows what these items are, we use a CNN to "see" and classify them based on the visual rendering.

The process consists of three main steps:
1. **Data Generation**: Creating a dataset of images.
2. **Training**: Teaching the neural network to recognize the patterns.
3. **Integration**: Using the trained model within the environment.

### Interactive Learning

You can follow this tutorial interactively using Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/adversarial2dEnvAI/blob/master/notebooks/CNN_Training.ipynb)

## 1. Data Generation

Before we can train a model, we need data. We use `src/custom_grid_env/cnn_tutorial/data_generation.py` to procedurally generate 64x64 pixel images.

### How it works:
- **Diverse Backgrounds**: To make the model robust, we generate items on white, red crosshatched, and green crosshatched backgrounds.
- **Randomness**: We add slight random offsets to the position of the dog or flower in each image.
- **Output**: Images are saved in the `data/dog/` and `data/flower/` directories.

To generate the data, run:
```bash
python src/custom_grid_env/cnn_tutorial/data_generation.py
```

## 2. Training the Neural Network

The training logic is located in `src/custom_grid_env/cnn_tutorial/train.py`. We use **TensorFlow** and **Keras** to build and train our model.

### The CNN Architecture
A CNN is designed to process pixel data. Our simple model uses the following layers:
- **Conv2D**: This layer "slides" filters over the image to detect features like edges or shapes.
- **MaxPooling2D**: This reduces the spatial size of the representation, making the model faster and more robust to small translations.
- **Flatten**: This turns the 2D feature maps into a 1D vector.
- **Dense**: Fully connected layers that perform the final classification. We use a 'softmax' activation at the end to get probabilities for each class (dog or flower).

### Key Concepts for Students:
- **Normalization**: We divide pixel values (0-255) by 255 to scale them to the range [0, 1]. This helps the neural network learn faster.
- **Train/Validation Split**: We set aside 20% of the data to test the model on images it hasn't seen during training. This tells us if the model is "overfitting" (memorizing) or actually learning.

To start training, run:
```bash
python src/custom_grid_env/cnn_tutorial/train.py
```

## 3. Evaluating Performance

After training, the script saves two plots in the `results/` directory:
- **training_metrics.png**: Shows how Accuracy and Loss changed over time. Ideally, Accuracy should go up and Loss should go down.
- **confusion_matrix.png**: Shows where the model made mistakes (e.g., how many flowers were misclassified as dogs).

## 4. Integration in the Environment

The trained model is saved as `model.keras`. The `PygameRenderer` in `src/custom_grid_env/renderer.py` automatically looks for this file.

When the agent stands on a cell containing a dog or flower, the renderer:
1. Captures a 64x64 "snapshot" of the cell.
2. Passes it to the CNN.
3. Displays the prediction and confidence score in the info panel.

---

### Exercises for Students:
1. **Modify the Data**: Change the `line_spacing` in `data_generation.py` and see how it affects training.
2. **Experiment with the Model**: Add another `Conv2D` layer in `train.py` and compare the accuracy.
3. **Hyperparameters**: Change the `epochs` or `batch_size` and observe the impact on the training graphs.
