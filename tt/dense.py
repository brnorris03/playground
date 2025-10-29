import numpy as np

def relu(x):
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)

def single_layer_dense_network(inputs, weights, bias):
    """
    Implements a single-layer dense neural network with ReLU activation.

    Args:
        inputs (numpy.ndarray): Input data, shape (num_samples, num_features).
        weights (numpy.ndarray): Weight matrix, shape (num_features, num_neurons).
        bias (numpy.ndarray): Bias vector, shape (num_neurons,).

    Returns:
        numpy.ndarray: Output of the network after ReLU activation.
    """
    # Linear transformation: Z = X * W + B
    linear_output = np.dot(inputs, weights) + bias

    # Apply ReLU activation
    activated_output = relu(linear_output)

    return activated_output

# Example Usage:
# Define input data, weights, and bias
num_samples = 4
num_features = 3
num_neurons = 2

# Randomly generated input data
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [-1.0, 0.5, 2.5],
              [0.0, -2.0, 1.0]])

# Randomly initialized weights
W = np.array([[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6]])

# Randomly initialized bias
B = np.array([0.05, -0.1])

# Perform the forward pass through the network
output = single_layer_dense_network(X, W, B)

print("Input data (X):\n", X)
print("\nWeights (W):\n", W)
print("\nBias (B):\n", B)
print("\nNetwork output (after ReLU activation):\n", output)
