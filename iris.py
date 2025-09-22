import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Importações dos seus arquivos
from perceptron import Perceptron
from util import plot_decision_regions

def split_data(X, y, test_size=0.3, random_state=None):
    """Divide os arrays X e y em subconjuntos de treino e teste."""
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(X.shape[0] * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

class MyStandardScaler:
    """Normaliza as features (z-score) para que tenham média 0 e desvio padrão 1."""
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
    def transform(self, X):
        std = np.where(self.std_ == 0, 1, self.std_)
        return (X - self.mean_) / std
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def calculate_accuracy(y_true, y_pred):
    """Calcula a acurácia comparando os valores verdadeiros com as predições."""
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

# Passo 1: Carregar e preparar o dataset
iris = datasets.load_iris()
mask = iris.target != 2
# Usando as features [0, 2] = comprimento da sépala e comprimento da pétala
X = iris.data[mask][:, [0, 2]] 
y = iris.target[mask]
print("=" * 50)
print("EXERCÍCIO 1: IRIS DATASET")
print("=" * 50)


# Passo 2: Dividir em treino/teste (70/30)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)


# Passo 3: Normalizar os dados
scaler = MyStandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)


# Passo 4: Treinar o perceptron
ppn = Perceptron(learning_rate=0.01, n_epochs=25)
ppn.fit(X_train_std, y_train)


# Passo 5: Calcular e reportar a acurácia
y_pred = ppn.predict(X_test_std)
accuracy = calculate_accuracy(y_test, y_pred)
print(f"\nAcurácia no conjunto de teste: {accuracy:.2%}")


# Passo 6: Plotar as regiões de decisão
print("\nGerando gráficos de resultado...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plt.sca(axes[0])
plot_decision_regions(X_train_std, y_train, classifier=ppn)
axes[0].set_title('Regiões de Decisão - Iris Dataset')
axes[0].set_xlabel('Comprimento da Sépala (normalizado)')
axes[0].set_ylabel('Comprimento da Pétala (normalizado)')
axes[0].legend(loc='upper left')

plt.sca(axes[1])
axes[1].plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, marker='o')
axes[1].set_title('Convergência do Treinamento')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('Número de erros de classificação')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""
A análise do Exercício 1 com o dataset
Iris demonstrou o cenário ideal de funcionamento do
Perceptron. Utilizando as features de comprimento da sépala
e da pétala, que são linearmente separáveis para
as classes Setosa e Versicolor, o modelo atingiu
a acurácia máxima de 100.00% no conjunto de
teste. Este desempenho perfeito foi alcançado com uma
rápida convergência em apenas 3 épocas, como evidenciado
pelo gráfico de 'Convergência do Treinamento', que mostra
os erros de classificação sendo zerados. O gráfico
de 'Regiões de Decisão' corrobora este resultado, exibindo
uma fronteira de decisão linear que separa as
duas classes de forma impecável. O experimento, portanto,
confirma na prática a teoria de que o
Perceptron é um algoritmo eficaz e garantido para
convergir em problemas de classificação binária com dados
linearmente separáveis.
"""