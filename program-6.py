import numpy as np

class MultiClassPerceptron:
    def __init__(self, input_size, num_classes, learning_rate=0.1, epochs=100):
        # Initialize weights for each class (one perceptron per class)
        self.weights = np.random.rand(num_classes, input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_classes = num_classes

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias to input
        outputs = []
        for weights in self.weights:
            weighted_sum = np.dot(weights, x)
            outputs.append(self.activation_function(weighted_sum))
        return np.argmax(outputs)  # Return the index of the highest output

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            total_error = 0
            for x, label in zip(training_inputs, labels):
                x = np.insert(x, 0, 1)  # Add bias to input
                for i in range(self.num_classes):
                    # Create target output: 1 for the correct class, 0 for others
                    target = 1 if label == i else 0
                    weighted_sum = np.dot(self.weights[i], x)
                    output = self.activation_function(weighted_sum)
                    error = target - output
                    total_error += abs(error)
                    # Update weights for each class
                    self.weights[i] += self.learning_rate * error * x
            if total_error == 0:
                print(f"Training completed in {epoch+1} epochs.")
                break

    def recall(self, test_inputs):
        predictions = [self.predict(x) for x in test_inputs]
        return predictions

# Example usage
if __name__ == "__main__":
    # Training data for a 3-class problem (e.g., basic OR-like behavior for each class)
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    labels = np.array([0, 1, 2, 1])  # Expected class labels

    # Initialize the multi-class perceptron with 3 output classes
    perceptron = MultiClassPerceptron(input_size=2, num_classes=3, learning_rate=0.1, epochs=100)

    # Train the perceptron
    perceptron.train(training_inputs, labels)

    # Testing (Recall) phase
    test_inputs = training_inputs  # Using the same inputs for testing
    predictions = perceptron.recall(test_inputs)

    print("Testing Results (Recall):")
    for inp, pred in zip(test_inputs, predictions):
        print(f"Input: {inp}, Predicted Class: {pred}")
