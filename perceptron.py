import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_history = []

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for epoch in range(self.n_epochs):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                error = y[idx] - y_predicted
                update = self.learning_rate * error
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0.0)
            self.errors_history.append(errors)
            if errors == 0:
                print(f"Convergiu na época {epoch + 1}")
                break
    
    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        return self.activation(self.net_input(X))