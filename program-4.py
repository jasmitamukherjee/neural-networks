import numpy as np

# Define the bipolar inputs for XOR function
inputs = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])

# Define the bipolar targets for XOR function (0 -> -1, 1 -> +1)
# XOR Truth Table in bipolar form:
# Input (-1, -1) -> Output -1
# Input (-1,  1) -> Output  1
# Input ( 1, -1) -> Output  1
# Input ( 1,  1) -> Output -1
targets = np.array([-1, 1, 1, -1])

# Initialize weights with small random values
weights = np.random.randn(2)

# Define learning rate
learning_rate = 0.1

# Hebbian learning function
def hebbian_learning(inputs, targets, weights, learning_rate, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for i in range(len(inputs)):
            x = inputs[i]
            y = targets[i]
            # Update weights using Hebbian rule: Δw = x * y * learning_rate
            weights += learning_rate * x * y
            print(f"Input: {x}, Target: {y}, Updated Weights: {weights}")
    return weights

# Train the XOR function using Hebbian learning
final_weights = hebbian_learning(inputs, targets, weights, learning_rate)

print("\nFinal weights after training:", final_weights)

# Test the trained model
print("\nTesting the trained model on XOR inputs:")
for i in range(len(inputs)):
    x = inputs[i]
    # Compute the output by taking the dot product of input and weights
    output = np.sign(np.dot(x, final_weights))
    print(f"Input: {x}, Output: {output}, Expected: {targets[i]}")
