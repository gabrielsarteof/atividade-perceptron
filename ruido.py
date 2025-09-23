import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Importações dos seus arquivos
from perceptron import Perceptron
from util import plot_decision_regions

# Pede o valor de N (paciência) para o usuário no início
patience_value = int(input("Digite o número de épocas para a paciência (N): "))

# --- Experimento 1: Variando a separação das classes ---
print("\n" + "=" * 50)
print("EXPERIMENTO 1: EFEITO DA SEPARAÇÃO (class_sep)")
print("=" * 50)
class_separation_values = [0.5, 1.0, 1.5, 2.0, 3.0]
for sep in class_separation_values:
    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=sep, flip_y=0.05, random_state=42
    )
    
    # Divide em treino+validação (70%) e teste (30%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # Divide o conjunto de 70% em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
    
    ppn = Perceptron(learning_rate=0.01, n_epochs=100, patience=patience_value)
    start_time = time.time()
    ppn.fit(X_train_std, y_train, X_val=X_val_std, y_val=y_val)
    end_time = time.time()
    training_time_ms = (end_time - start_time) * 1000
    
    y_pred = ppn.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- Resultado para class_sep = {sep:.1f} ---")
    print(f"Acurácia: {accuracy:.2%} | Tempo: {training_time_ms:.2f} ms")
    print("Matriz de Confusão:")
    print(conf_matrix)

# --- Experimento 2: Variando o ruído nos rótulos ---
print("\n" + "=" * 50)
print("EXPERIMENTO 2: EFEITO DO RUÍDO NOS RÓTULOS (flip_y)")
print("=" * 50)
flip_y_values = [0.0, 0.05, 0.1, 0.2]
for flip in flip_y_values:
    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=1.5, flip_y=flip, random_state=42
    )
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)
    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
    
    ppn = Perceptron(learning_rate=0.01, n_epochs=100, patience=patience_value)
    start_time = time.time()
    ppn.fit(X_train_std, y_train, X_val=X_val_std, y_val=y_val)
    end_time = time.time()
    training_time_ms = (end_time - start_time) * 1000
    
    y_pred = ppn.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n--- Resultado para flip_y = {flip:.2f} ---")
    print(f"Acurácia: {accuracy:.2%} | Tempo: {training_time_ms:.2f} ms")
    print("Matriz de Confusão:")
    print(conf_matrix)

# --- Plotando um exemplo para visualização ---
print("\nGerando gráficos de exemplo (class_sep=1.5, flip_y=0.05)...")
X_example, y_example = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, class_sep=1.5, flip_y=0.05, random_state=42
)
X_train_full_ex, X_test_ex, y_train_full_ex, y_test_ex = train_test_split(X_example, y_example, test_size=0.3, random_state=42, stratify=y_example)
X_train_ex, X_val_ex, y_train_ex, y_val_ex = train_test_split(X_train_full_ex, y_train_full_ex, test_size=0.2, random_state=42, stratify=y_train_full_ex)
scaler_ex = StandardScaler()
X_train_std_ex = scaler_ex.fit_transform(X_train_ex)
X_test_std_ex = scaler_ex.transform(X_test_ex)
X_val_std_ex = scaler_ex.transform(X_val_ex)

ppn_ex = Perceptron(learning_rate=0.01, n_epochs=100, patience=patience_value)
ppn_ex.fit(X_train_std_ex, y_train_ex, X_val=X_val_std_ex, y_val=y_val_ex)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plt.sca(axes[0])
plot_decision_regions(X_train_std_ex, y_train_ex, classifier=ppn_ex)
axes[0].set_title('Visualização do Dataset (Exemplo)')
axes[0].set_xlabel('Feature 1 (normalizada)')
axes[0].set_ylabel('Feature 2 (normalizada)')
axes[0].legend(loc='upper left')

plt.sca(axes[1])
axes[1].plot(range(1, len(ppn_ex.errors_history) + 1), ppn_ex.errors_history, marker='o')
axes[1].set_title('Convergência do Treinamento (Exemplo)')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('Número de erros de classificação')
axes[1].grid(True, alpha=0.3)
<<<<<<< HEAD
plt.savefig('ruido_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
=======

>>>>>>> 9cd4a44d02cbaab03c43523a6a9798c605639dcc
plt.tight_layout()
plt.show()

"""
ANÁLISE DO RELATÓRIO FINAL - EXERCÍCIO 4

Este exercício testou a sensibilidade do Perceptron
à qualidade dos dados. No primeiro experimento, a
acurácia foi diretamente proporcional à separação das
classes (`class_sep`), subindo de 36.67% (baixa separação)
para um pico de 96.67% (alta separação). A análise
das matrizes de confusão confirmou essa tendência,
mostrando que o número de erros diminuiu drasticamente
com classes mais distintas.

No segundo experimento, a relação com o ruído (`flip_y`)
foi mais complexa. A acurácia foi de 80.00% com 0%
de ruído, atingiu um pico de 93.33% com 5% de ruído,
e depois caiu para 63.33% com 20% de ruído. Isso sugere
que um pequeno ruído pode ter ajudado o modelo a
generalizar melhor, mas níveis altos prejudicaram o desempenho.

A técnica de parada antecipada com paciência N=10
foi ativada em todos os testes, interrompendo o
treino (entre as épocas 11 e 23) e mantendo o
tempo de treinamento baixo (entre 12 e 26 ms).
Isso demonstrou a eficácia do método para otimizar
o treino em cenários com dados imperfeitos.
"""

#finalizado