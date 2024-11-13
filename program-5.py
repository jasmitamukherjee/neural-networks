import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(input_size + 1)  # +1 for the bias weight
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Add bias input
        weighted_sum = np.dot(self.weights, x)
        return self.activation_function(weighted_sum)

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            total_error = 0
            for x, label in zip(training_inputs, labels):
                prediction = self.predict(x)
                error = label - prediction
                total_error += abs(error)
                self.weights += self.learning_rate * error * np.insert(x, 0, 1)  # Update weights
            if total_error == 0:
                print(f"Training completed in {epoch+1} epochs.")
                break

    def recall(self, test_inputs):
        predictions = [self.predict(x) for x in test_inputs]
        return predictions

# Example usage
if __name__ == "__main__":
    # Define training data for an AND gate
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    labels = np.array([0, 0, 0, 1])  # AND gate output

    # Initialize the Perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=100)

    # Train the Perceptron
    perceptron.train(training_inputs, labels)

    # Testing the Perceptron (Recall phase)
    test_inputs = training_inputs  # Using same inputs for testing
    predictions = perceptron.recall(test_inputs)
    
    print("Testing Results (Recall):")
    for inp, pred in zip(test_inputs, predictions):
        print(f"Input: {inp}, Prediction: {pred}")
