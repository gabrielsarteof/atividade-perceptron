import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
from sklearn.metrics import confusion_matrix # matriz de confusão

# Importações dos arquivos
from perceptron import Perceptron, calculate_accuracy
from util import plot_decision_regions

def split_data(X, y, test_size=0.3, random_state=None):
    """
    Função para dividir os arrays X e y em subconjuntos de treino e teste.
    """
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(X.shape[0] * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    # Correção da atribuição das variáveis
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

class MyStandardScaler:
    """
    Classe para normalizar as features (z-score) para que tenham média 0 e desvio padrão 1.
    """
    def fit(self, X):
        """Método para calcular a média e o desvio padrão dos dados de treino."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        """Método para aplicar a normalização usando os valores já calculados."""
        std = np.where(self.std_ == 0, 1, self.std_)
        return (X - self.mean_) / std

    def fit_transform(self, X):
        """Método para executar o fit e o transform em sequência."""
        self.fit(X)
        return self.transform(X)

# Passo 1: Carregar e preparar o dataset
iris = datasets.load_iris()
mask = iris.target != 2
X = iris.data[mask][:, [0, 2]] 
y = iris.target[mask]
print("=" * 50)
print("EXERCÍCIO 1: IRIS DATASET")
print("=" * 50)
print(f"Número de Amostras: {X.shape[0]}")
print(f"Features Usadas: {iris.feature_names[0]} e {iris.feature_names[2]}")

# Passo 2: Dividir em treino/teste (70/30)
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)

# Passo 3: Normalizar os dados
scaler = MyStandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# Passo 4: Treinar o perceptron
ppn = Perceptron(learning_rate=0.01, n_epochs=25)
start_time = time.time()
ppn.fit(X_train_std, y_train)
end_time = time.time()
training_time_ms = (end_time - start_time) * 1000

# Passo 5: Calcular e reportar a acurácia
y_pred = ppn.predict(X_test_std)
accuracy = calculate_accuracy(y_test, y_pred)
print(f"\nAcurácia no conjunto de teste: {accuracy:.2%}")
print(f"Tempo de Treinamento: {training_time_ms:.2f} ms")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Adicionado: Bloco de Análise Detalhada
print(f"- Erros finais no treino: {ppn.errors_history[-1]}")
if 0 in ppn.errors_history:
    conv_epoch = ppn.errors_history.index(0) + 1
    print(f"- Convergiu na época: {conv_epoch}")
else:
    print("- Não convergiu completamente")

print(f"\nPesos aprendidos:")
print(f"- w1 ({iris.feature_names[0]}): {ppn.weights[0]:.4f}")
print(f"- w2 ({iris.feature_names[2]}): {ppn.weights[1]:.4f}")
print(f"- bias: {ppn.bias:.4f}")

if ppn.weights[1] != 0:
    slope = -ppn.weights[0] / ppn.weights[1]
    intercept = -ppn.bias / ppn.weights[1]
    print(f"\nEquação da fronteira de decisão:")
    print(f"x2 = {slope:.2f} * x1 + {intercept:.2f}")
# Fim do Bloco

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
plt.savefig('iris_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.tight_layout()
plt.show()

"""
ANÁLISE DO RELATÓRIO FINAL - EXERCÍCIO 1

A análise do Exercício 1 com o dataset Iris
demonstrou o cenário ideal de funcionamento do Perceptron.
Utilizando as features 'sepal length (cm)' e 'petal length (cm)',
que são linearmente separáveis para as classes Setosa
e Versicolor, o modelo atingiu a acurácia máxima
de 100.00% no conjunto de teste. Este desempenho
perfeito foi alcançado com uma rápida convergência em
apenas 3 épocas e um tempo de treinamento de
1.98 ms. A Matriz de Confusão, com resultado
de [[17, 0], [0, 13]], confirma a ausência
total de erros (0 falsos positivos e 0 falsos
negativos). O gráfico de 'Regiões de Decisão' corrobora
este resultado, exibindo uma fronteira de decisão linear
que separa as duas classes de forma impecável.
O experimento, portanto, confirma na prática a teoria
de que o Perceptron é um algoritmo eficaz e
garantido para convergir em problemas de classificação
binária com dados linearmente separáveis.
"""

#finalizado