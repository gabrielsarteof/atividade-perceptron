import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from perceptron import Perceptron
from util import plot_decision_regions

print("=" * 50)
print("EXEMPLO: BLOBS SINTÉTICOS")
print("=" * 50)


X, y = datasets.make_blobs(
    n_samples=200, 
    n_features=2, 
    centers=2, 
    cluster_std=1.5, 
    center_box=(-5, 5), 
    random_state=42 
)

print(f"Dataset gerado:")
print(f"- Amostras: {X.shape[0]}")
print(f"- Features: {X.shape[1]}")
print(f"- Classes: {np.unique(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, 
    random_state=42,
    stratify=y 
)

print(f"\nDivisão treino/teste:")
print(f"- Treino: {len(X_train)} amostras")
print(f"- Teste: {len(X_test)} amostras")

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train) 
X_test_std = scaler.transform(X_test) 

ppn = Perceptron(learning_rate=0.01, n_epochs=50)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nResultados:")
print(f"- Acurácia: {accuracy:.2%}")
print(f"- Erros finais no treino: {ppn.errors_history[-1]}")


if 0 in ppn.errors_history:
    conv_epoch = ppn.errors_history.index(0)
    print(f"- Convergiu na época: {conv_epoch + 1}")
else:
    print("- Não convergiu completamente")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))


axes[0].set_title('Regiões de Decisão - Blobs')
plot_decision_regions(X_train_std, y_train, classifier=ppn)
axes[0].set_xlabel('Feature 1 (normalizada)')
axes[0].set_ylabel('Feature 2 (normalizada)')


axes[1].plot(range(1, len(ppn.errors_history) + 1), ppn.errors_history, marker='o')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('Número de erros')
axes[1].set_title('Convergência do Treinamento')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


print(f"\nPesos aprendidos:")
print(f"- w1: {ppn.weights[0]:.4f}")
print(f"- w2: {ppn.weights[1]:.4f}")
print(f"- bias: {ppn.bias:.4f}")


if ppn.weights[1] != 0:
    slope = -ppn.weights[0]/ppn.weights[1]
    intercept = -ppn.bias/ppn.weights[1]
    print(f"\nEquação da fronteira de decisão:")
    print(f"x2 = {slope:.2f} * x1 + {intercept:.2f}")