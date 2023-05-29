# SGCNet Node Classification on Cora Dataset

This repository contains code that demonstrates node classification using SGCNet (Simplifying Graph Convolutional Networks) on the Cora dataset. The code utilizes PyTorch and torch-geometric libraries to implement the SGCNet model and evaluate its performance.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Functions](#functions)
- [Results](#results)

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3
- PyTorch
- torch-geometric
- numpy
- matplotlib

## Usage

1. Set the `path` variable to the directory path where the dataset is located.

```python
path = "drive/MyDrive"
```

2. Load the Cora dataset using the `Planetoid` class from `torch-geometric`. The dataset will be stored in the `dataset` variable.

```python
dataset = Planetoid(path, "Cora")
```

3. Retrieve the first graph from the dataset and store it in the `data` variable.

```python
data = dataset[0]
```

4. Print information about the Cora dataset.

```python
print('Cora:', data)
```

## Functions

The code defines two functions:

### `train_SGCNet()`

This function trains the SGCNet model using the loaded dataset. It performs the following steps:

- Sets the model to training mode.
- Zeros out the gradients of the optimizer.
- Passes the input through the model to get the predicted output.
- Computes the loss using the negative log-likelihood loss function.
- Performs backpropagation to compute the gradients.
- Updates the model parameters using the optimizer.
- Returns the predicted output.

### `test_SGCNet()`

This function evaluates the SGCNet model's performance on the loaded dataset. It performs the following steps:

- Sets the model to evaluation mode.
- Computes the logits using the model.
- Calculates the accuracy for each split of the dataset (train, validation, test).
- Returns a list of accuracies.

## Results

The code runs a training loop for a specified number of epochs. In each epoch, it trains the SGCNet model using the `train_SGCNet()` function and evaluates its performance using the `test_SGCNet()` function. The best validation accuracy and corresponding test accuracy are tracked and printed for each epoch.

After the training loop, the code generates a scatter plot to visualize the predicted node embeddings. The plot displays the embeddings for each class separately.

Feel free to modify the code and experiment with different parameters to explore the performance of SGCNet on the Cora dataset.
