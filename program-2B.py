# Define weights for AND-NOT logic
weights = [1, -1]  # A has weight 1, B has weight -1
threshold = 1      # Threshold for AND-NOT function

# Define the AND-NOT logic inputs and expected outputs
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
expected_outputs = [0, 0, 1, 0]  # AND-NOT truth table

# McCulloch-Pitts neuron function for AND-NOT logic
def mc_pitts_andnot(inputs, weights, threshold):
    # Calculate weighted sum
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    # Apply threshold function
    return 1 if weighted_sum >= threshold else 0

# Test the neuron with all input combinations
print("AND-NOT Logic using McCulloch-Pitts Neuron Model:")
for i, input_pair in enumerate(inputs):
    output = mc_pitts_andnot(input_pair, weights, threshold)
    print(f"Input: {input_pair}, Output: {output}, Expected: {expected_outputs[i]}")
