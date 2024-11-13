# Define the weights for OR logic
weights = [1, 1]  # Each input contributes equally for OR
threshold = 1     # Threshold to activate the neuron

# Define the OR logic inputs and expected outputs
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
expected_outputs = [0, 1, 1, 1]  # OR logic truth table

# McCulloch-Pitts neuron function for OR logic
def mc_pitts_neuron(inputs, weights, threshold):
    # Calculate weighted sum
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    # Apply threshold function
    return 1 if weighted_sum >= threshold else 0

# Test the neuron with all input combinations
print("OR Logic using McCulloch-Pitts Neuron Model:")
for i, input_pair in enumerate(inputs):
    output = mc_pitts_neuron(input_pair, weights, threshold)
    print(f"Input: {input_pair}, Output: {output}, Expected: {expected_outputs[i]}")
