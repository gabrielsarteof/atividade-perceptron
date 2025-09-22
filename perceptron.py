import numpy as np

# A função de acurácia é definida aqui
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=100, patience=float('inf')):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_history = []
        self.patience = patience

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        best_val_accuracy = 0.0
        epochs_no_improve = 0
        
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
            
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                current_val_accuracy = calculate_accuracy(y_val, val_pred)
                
                if current_val_accuracy > best_val_accuracy:
                    best_val_accuracy = current_val_accuracy
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= self.patience:
                    print(f"\nParada antecipada na época {epoch + 1}!")
                    break

            if errors == 0:
                print(f"Convergiu na época {epoch + 1} (erro de treino zero)")
                break
    
    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        return self.activation(self.net_input(X))
    
    
#finalizado