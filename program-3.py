import numpy as np

# 1. Binary Unipolar Activation Function (0 or 1)
def binary_unipolar(x, threshold=0):
    return 1 if x >= threshold else 0

# 2. Binary Bipolar Activation Function (-1 or 1)
def binary_bipolar(x, threshold=0):
    return 1 if x >= threshold else -1

# 3. Continuous Unipolar Activation Function (0 to 1)
def continuous_unipolar(x):
    return 1 / (1 + np.exp(-x))  # Similar to Sigmoid, range (0, 1)

# 4. Sigmoid Activation Function (0 to 1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 5. ReLU Activation Function (0 to ∞)
def relu(x):
    return max(0, x)

# Testing the activation functions with sample inputs
inputs = [-2, -1, 0, 1, 2]

print("Binary Unipolar Activation:")
for x in inputs:
    print(f"Input: {x}, Output: {binary_unipolar(x)}")

print("\nBinary Bipolar Activation:")
for x in inputs:
    print(f"Input: {x}, Output: {binary_bipolar(x)}")

print("\nContinuous Unipolar Activation:")
for x in inputs:
    print(f"Input: {x}, Output: {continuous_unipolar(x)}")

print("\nSigmoid Activation:")
for x in inputs:
    print(f"Input: {x}, Output: {sigmoid(x)}")

print("\nReLU Activation:")
for x in inputs:
    print(f"Input: {x}, Output: {relu(x)}")
