# Fashion MNIST Classification with TensorFlow and tf.keras

## Overview
This repository contains a simple example of image classification using TensorFlow and the tf.keras library. In this example, we'll use the Fashion MNIST dataset, which consists of 70,000 grayscale images in 10 categories. The goal is to train a neural network to accurately classify these images into their respective categories.

### Prerequisites
Before running the code, make sure you have TensorFlow installed. You can install it using pip:

```bash
pip install tensorflow
```

## Dataset
We'll be using the Fashion MNIST dataset, which is available directly through TensorFlow. This dataset contains 60,000 training images and 10,000 test images, each of size 28x28 pixels. Here are the categories represented in the dataset:

| Label | Class         |
|-------|---------------|
| 0     | T-shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

### Loading the Dataset
We can load the Fashion MNIST dataset using the following code:

```python
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

### Data Preprocessing
Before feeding the data into the neural network, we need to preprocess it. We scale the pixel values from the range [0, 255] to [0, 1]:

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

## Building the Model
We'll build a simple neural network model with two layers using the Sequential API in TensorFlow:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```

The first layer flattens the 28x28 pixel input images into a one-dimensional array. The second layer is a dense layer with 128 neurons and ReLU activation, and the final layer has 10 neurons representing the 10 classes.

## Compiling the Model
Before training, we compile the model by specifying the loss function, optimizer, and metrics:

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

## Training the Model
To train the model, we use the training data and labels and specify the number of epochs:

```python
model.fit(train_images, train_labels, epochs=10)
```

## Evaluating the Model
We evaluate the model's accuracy on the test dataset:

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```

## Making Predictions
We can use the trained model to make predictions on new images. Here's how to do it:

```python
import numpy as np

# Get a sample image from the test dataset
img = test_images[0]

# Expand dimensions to create a batch with a single image
img = np.expand_dims(img, 0)

# Make predictions
predictions = model.predict(img)

# Find the label with the highest confidence
predicted_label = np.argmax(predictions[0])
```

## Visualizing Predictions
To visualize predictions, you can use the provided functions `plot_image` and `plot_value_array`:

```python
import matplotlib.pyplot as plt

def plot_image(i, predictions_array, true_label, img):
    # ... (code for plotting image)

def plot_value_array(i, predictions_array, true_label):
    # ... (code for plotting value array)
```

You can then visualize predictions for multiple images as shown in the example code.

This README provides an overview of the code and its functionality. You can run the provided code in your Python environment to train a model for Fashion MNIST classification. Feel free to customize and extend the code for your own projects.
