# Define weights and threshold for AND, OR, and NOT gates
and_weights = [1, 1]
or_weights = [1, 1]
not_weight = -1
threshold_and = 2
threshold_or = 1
threshold_not = 0

# Define the XOR logic inputs and expected outputs
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
expected_outputs = [0, 1, 1, 0]  # XOR logic truth table

# McCulloch-Pitts neuron function
def mc_pitts_neuron(inputs, weights, threshold):
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    return 1 if weighted_sum >= threshold else 0

# NOT function for a single input
def mc_pitts_not(input_value):
    return mc_pitts_neuron([input_value], [not_weight], threshold_not)

# XOR function using McCulloch-Pitts neurons
def mc_pitts_xor(a, b):
    # Layer 1: Calculate A AND (NOT B) and (NOT A) AND B
    not_b = mc_pitts_not(b)
    not_a = mc_pitts_not(a)
    
    and1 = mc_pitts_neuron([a, not_b], and_weights, threshold_and)
    and2 = mc_pitts_neuron([not_a, b], and_weights, threshold_and)
    
    # Layer 2: OR the results of the two AND operations
    xor_result = mc_pitts_neuron([and1, and2], or_weights, threshold_or)
    return xor_result

# Test the XOR neuron with all input combinations
print("XOR Logic using McCulloch-Pitts Neuron Model:")
for i, (a, b) in enumerate(inputs):
    output = mc_pitts_xor(a, b)
    print(f"Input: ({a}, {b}), Output: {output}, Expected: {expected_outputs[i]}")
