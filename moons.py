import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
from sklearn.metrics import confusion_matrix

# Importações dos arquivos
from perceptron import Perceptron, calculate_accuracy
from util import plot_decision_regions

def split_data(X, y, test_size=0.3, random_state=None):
    """
    Função para dividir os arrays X e y em subconjuntos de treino e teste.
    """
    if random_state is not None:
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

# Adicionado: Pede o valor de N (paciência) para o usuário
patience_value = int(input("Digite o número de épocas para a paciência (N): "))

# Passo 1: Gerar o dataset
X, y = datasets.make_moons(n_samples=200, noise=0.15, random_state=42)
print("=" * 50)
print("EXERCÍCIO 2: MOONS DATASET")
print("=" * 50)
print(f"Número de Amostras: {X.shape[0]}")
print(f"Número de Features: {X.shape[1]}")

# Passo 2: Dividir em treino, validação e teste
X_train_full, X_test, y_train_full, y_test = split_data(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = split_data(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Passo 3: Normalizar os dados
scaler = MyStandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

# Passo 4: Treinar o perceptron
# Usa o valor de paciência fornecido pelo usuário ( A análise será feita com 10 um número escolhido como o recomendado(nem tão pouco, nem muito))
ppn = Perceptron(learning_rate=0.01, n_epochs=50, patience=patience_value)
start_time = time.time()
ppn.fit(X_train_std, y_train, X_val=X_val_std, y_val=y_val)
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
print(f"- w1 (Feature 1): {ppn.weights[0]:.4f}")
print(f"- w2 (Feature 2): {ppn.weights[1]:.4f}")
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
axes[0].set_title('Regiões de Decisão - Moons Dataset')
axes[0].set_xlabel('Feature 1 (normalizada)')
axes[0].set_ylabel('Feature 2 (normalizada)')
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
ANÁLISE DO RELATÓRIO FINAL - EXERCÍCIO 2

O Exercício 2, utilizando o Moons dataset, cumpriu
com sucesso seu objetivo de demonstrar as limitações
fundamentais do Perceptron. Como os dados não são
linearmente separáveis, o algoritmo foi incapaz de convergir,
fato comprovado pelo gráfico de 'Convergência do Treinamento',
que exibe uma oscilação constante no número de erros
ao longo das 50 épocas. O modelo não foi interrompido
pela parada antecipada, pois a acurácia de validação
provavelmente oscilou sem uma tendência clara de melhora.

Apesar da falha de convergência, o modelo alcançou
uma acurácia de 80.00% no teste em apenas 2.01 ms.
A Matriz de Confusão ([[22, 5], [7, 26]]) mostra
que o modelo cometeu 12 erros no total. O gráfico de
'Regiões de Decisão' explica este resultado: a fronteira
linear encontrada representa o 'melhor esforço' do
Perceptron, classificando corretamente uma porção significativa
dos dados, mas falhando visivelmente em capturar a
estrutura curva das 'luas'. Este resultado evidencia que,
embora o Perceptron possa encontrar uma solução linear
razoável, ele é inerentemente inadequado para problemas
complexos que exigem fronteiras de decisão não-lineares.
"""

#finalizado