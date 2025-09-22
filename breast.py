import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
import time

# Importações dos seus arquivos
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

# Adicionado: Pede o valor de N (paciência) para o usuário no início
patience_value = int(input("Digite o número de épocas para a paciência (N): "))

# --------------------------------------------------------------------------
# --- VERSÃO A: USANDO APENAS 2 FEATURES ---
# --------------------------------------------------------------------------

print("\n" + "=" * 50)
print("EXERCÍCIO 3: BREAST CANCER - VERSÃO A (2 FEATURES)")
print("=" * 50)

# Passo 1: Carregar o dataset
cancer = datasets.load_breast_cancer()
X = cancer.data[:, [0, 1]]
y = cancer.target
print(f"Número de Amostras: {X.shape[0]}")
print(f"Features Usadas: {cancer.feature_names[0]} e {cancer.feature_names[1]}")

# Passo 2: Dividir em treino, validação e teste
X_train_full_A, X_test_A, y_train_full_A, y_test_A = split_data(X, y, test_size=0.3, random_state=42)
X_train_A, X_val_A, y_train_A, y_val_A = split_data(X_train_full_A, y_train_full_A, test_size=0.2, random_state=42)

# Passo 3: Normalizar os dados
scaler_A = MyStandardScaler()
scaler_A.fit(X_train_A)
X_train_std_A = scaler_A.transform(X_train_A)
X_val_std_A = scaler_A.transform(X_val_A)
X_test_std_A = scaler_A.transform(X_test_A)

# Passo 4: Treinar o perceptron
ppn_A = Perceptron(learning_rate=0.01, n_epochs=50, patience=patience_value)
start_time_A = time.time()
ppn_A.fit(X_train_std_A, y_train_A, X_val=X_val_std_A, y_val=y_val_A)
end_time_A = time.time()
training_time_ms_A = (end_time_A - start_time_A) * 1000

# Passo 5: Avaliar o modelo
y_pred_A = ppn_A.predict(X_test_std_A)
accuracy_A = calculate_accuracy(y_test_A, y_pred_A)
print(f"\nAcurácia (2 features): {accuracy_A:.2%}")
print(f"Tempo de Treinamento (2 features): {training_time_ms_A:.2f} ms")
print("\nRelatório de Classificação (2 features):")
print(classification_report(y_test_A, y_pred_A, target_names=cancer.target_names))
print("\nMatriz de Confusão (2 features):")
print(confusion_matrix(y_test_A, y_pred_A))

# Passo 6: Plotar os resultados
print("\nGerando gráficos para a Versão A (2 features)...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.sca(axes[0])
plot_decision_regions(X_train_std_A, y_train_A, classifier=ppn_A)
axes[0].set_title('Regiões de Decisão (2 Features)')
axes[0].set_xlabel(f'{cancer.feature_names[0]} (normalizado)')
axes[0].set_ylabel(f'{cancer.feature_names[1]} (normalizado)')
axes[0].legend(loc='upper left')
plt.sca(axes[1])
axes[1].plot(range(1, len(ppn_A.errors_history) + 1), ppn_A.errors_history, marker='o')
axes[1].set_title('Convergência (2 Features)')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('Número de erros de classificação')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# --- VERSÃO B: USANDO TODAS AS 30 FEATURES ---
# --------------------------------------------------------------------------

print("\n" + "=" * 50)
print("EXERCÍCIO 3: BREAST CANCER - VERSÃO B (30 FEATURES)")
print("=" * 50)

# Passo 1: Usar o dataset completo
X_full = cancer.data
y_full = cancer.target
print(f"Número de Amostras: {X_full.shape[0]}")
print(f"Número de Features: {X_full.shape[1]}")

# Passo 2: Dividir em treino, validação e teste
X_train_full_B, X_test_B, y_train_full_B, y_test_B = split_data(X_full, y_full, test_size=0.3, random_state=42)
X_train_B, X_val_B, y_train_B, y_val_B = split_data(X_train_full_B, y_train_full_B, test_size=0.2, random_state=42)

# Passo 3: Normalizar os dados
scaler_B = MyStandardScaler()
scaler_B.fit(X_train_B)
X_train_std_B = scaler_B.transform(X_train_B)
X_val_std_B = scaler_B.transform(X_val_B)
X_test_std_B = scaler_B.transform(X_test_B)

# Passo 4: Treinar o perceptron
ppn_B = Perceptron(learning_rate=0.01, n_epochs=50, patience=patience_value)
start_time_B = time.time()
ppn_B.fit(X_train_std_B, y_train_B, X_val=X_val_std_B, y_val=y_val_B)
end_time_B = time.time()
training_time_ms_B = (end_time_B - start_time_B) * 1000

# Passo 5: Avaliar o modelo
y_pred_B = ppn_B.predict(X_test_std_B)
accuracy_B = calculate_accuracy(y_test_B, y_pred_B)
print(f"\nAcurácia (30 features): {accuracy_B:.2%}")
print(f"Tempo de Treinamento (30 features): {training_time_ms_B:.2f} ms")
print("\nRelatório de Classificação (30 features):")
print(classification_report(y_test_B, y_pred_B, target_names=cancer.target_names))
print("\nMatriz de Confusão (30 features):")
print(confusion_matrix(y_test_B, y_pred_B))

# Passo 6: Plotar a convergência
print("\nGerando gráfico de convergência para a Versão B (30 features)...")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(1, len(ppn_B.errors_history) + 1), ppn_B.errors_history, marker='o')
ax.set_title('Convergência (30 Features)')
ax.set_xlabel('Épocas')
ax.set_ylabel('Número de erros de classificação')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""
ANÁLISE DO RELATÓRIO FINAL - EXERCÍCIO 3

A análise comparativa do Exercício 3 demonstra o
claro benefício de usar mais informações para o modelo.
A Versão B, utilizando todas as 30 features, alcançou
uma acurácia superior de 95.88%, contra 90.59% da
Versão A, que usou apenas 2 features. Ambos os
modelos utilizaram a parada antecipada (early stopping), com
o treino sendo interrompido nas épocas 14 e 15,
respectivamente, confirmando que os dados não são perfeitamente
linearmente separáveis. A superioridade da Versão B é
mais evidente na análise da Matriz de Confusão.
O modelo com 30 features reduziu os erros mais
críticos: os Falsos Negativos (câncer real diagnosticado
como benigno) caíram de 7 para apenas 2, e os
Falsos Positivos caíram de 9 para 5. Isso prova
que, para este problema, mais features resultaram em
um modelo não apenas mais preciso, mas significativamente
mais confiável e seguro para uma aplicação médica,
cumprindo o objetivo do exercício.
"""

#finalizado