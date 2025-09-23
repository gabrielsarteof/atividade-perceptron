# Implementação do Perceptron com Datasets Clássicos

## Descrição do Projeto

Este projeto implementa o algoritmo Perceptron do zero, aplicando-o em diversos datasets clássicos de machine learning para demonstrar suas capacidades e limitações. O Perceptron é um classificador binário linear fundamental, proposto por Frank Rosenblatt em 1957, que forma a base para o entendimento de redes neurais mais complexas.

## Dupla
- Gabriel Barbosa Sarte
- Tracy Julie Calabrez

## Estrutura do Projeto

```
atividade-perceptron/
├── perceptron.py              # Implementação da classe Perceptron
├── util.py                    # Funções de visualização (plot_decision_regions)
├── iris.py                    # Exercício 1: Iris Dataset
├── moons.py                   # Exercício 2: Moons Dataset
├── breast.py                  # Exercício 3: Breast Cancer Dataset
├── ruido.py                   # Exercício 4: Dataset com Ruído
├── dlsp.py                    # Exercício 5: Dataset Linearmente Separável Personalizado
├── bobs.py                    # Exemplo: Blobs Sintéticos
├── report_generator.py        # Gerador automático de relatórios HTML
├── quick_report.py            # Script de execução rápida
├── requirements.txt           # Dependências do projeto
├── README.md                  # Este arquivo
└── RELATORIO.md              # Análise acadêmica completa
```

## Como Executar

### Requisitos

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy
matplotlib
scikit-learn
```

### Execução dos Experimentos

#### Opção 1: Execução Rápida (Recomendada)
```bash
python quick_report.py
```
Este comando executa todos os experimentos automaticamente e abre o relatório HTML no navegador.

#### Opção 2: Execução Individual dos Scripts
```bash
# Exercício 1: Iris Dataset
python iris.py

# Exercício 2: Moons Dataset (solicita valor de paciência)
python moons.py

# Exercício 3: Breast Cancer Dataset (solicita valor de paciência)
python breast.py

# Exercício 4: Dataset com Ruído (solicita valor de paciência)
python ruido.py

# Exercício 5: Dataset Personalizado (solicita valor de paciência)
python dlsp.py

# Exemplo: Blobs Sintéticos
python bobs.py
```

**Observação:** Alguns scripts solicitam um valor de paciência (N) para early stopping. Recomenda-se usar N=10.

#### Opção 3: Geração Manual de Relatório
```bash
python report_generator.py
```
Este script executará todos os experimentos automaticamente e gerará um relatório HTML completo.

### Visualização dos Resultados

Para visualizar o relatório HTML gerado, você pode usar:

```bash
# Servidor Python simples
python -m http.server 8000
# Acesse http://localhost:8000 e clique no arquivo HTML

# Ou usando Node.js serve (se instalado)
npm install -g serve
serve .
```

## Características Específicas da Implementação

### Funcionalidades Implementadas
1. **Early Stopping:** Implementado com paciência configurável para evitar overfitting
2. **Normalização Personalizada:** Classe `MyStandardScaler` implementada do zero
3. **Divisão de Dados:** Função `split_data()` personalizada para treino/teste
4. **Visualização:** Função `plot_decision_regions()` para visualizar fronteiras de decisão
5. **Métricas Detalhadas:** Função `calculate_accuracy()` e análise de matriz de confusão

### Estrutura da Classe Perceptron
```python
class Perceptron:
    def __init__(self, learning_rate=0.1, n_epochs=100, patience=float('inf'))
    def activation(self, x)           # Função step
    def fit(self, X, y, X_val, y_val) # Treinamento com validação
    def net_input(self, X)            # Entrada líquida
    def predict(self, X)              # Predições
```

### Configurações Padrão
- **Learning Rate:** 0.01 (testado e otimizado)
- **Épocas máximas:** 25-100 (varia por experimento)
- **Paciência:** Configurável pelo usuário (recomendado: 10)
- **Resolução de visualização:** 0.02

## Arquivos Gerados

### Gráficos
Os scripts geram automaticamente arquivos PNG:
- `iris_chart.png` - Regiões de decisão e convergência (Iris)
- `moons_chart.png` - Limitações com dados não-lineares (Moons)
- `breast_chart.png` - Comparação 2D vs 30D (Cancer)
- `ruido_chart.png` - Efeitos de separação e ruído
- `dlsp_chart.png` - Dataset personalizado com análise completa

### Relatório HTML
- `relatorio_perceptron_YYYYMMDD_HHMMSS.html` - Relatório completo gerado automaticamente

## Report Generator

O `report_generator.py` é uma ferramenta que:

1. **Executa todos os scripts automaticamente** com tratamento de erro
2. **Extrai métricas** dos outputs usando regex
3. **Carrega gráficos** salvos pelos scripts
4. **Gera relatório HTML profissional** com:
   - Navegação interativa
   - KPIs consolidados
   - Matrizes de confusão
   - Análises comparativas
   - Gráficos incorporados em base64

## Execução Recomendada

Para a melhor experiência, siga esta sequência:

1. **Ambiente virtual:**
```bash
python -m venv perceptron_env
source perceptron_env/bin/activate  # Linux/Mac
# ou
perceptron_env\Scripts\activate     # Windows
pip install -r requirements.txt
```

2. **Execução completa:**
```bash
python quick_report.py
```

3. **Teste individual (opcional):**
```bash
python iris.py        # Exemplo simples
python bobs.py         # Blobs sintéticos
```

## Estrutura dos Resultados

### Arquivo HTML Gerado
O relatório HTML contém:
- **Executive Summary** com KPIs consolidados
- **Navegação interativa** entre seções
- **Análises detalhadas** de cada experimento
- **Matrizes de confusão** formatadas
- **Gráficos incorporados** em alta resolução
- **Comparações lado a lado** dos resultados
- **Design responsivo** para diferentes dispositivos

### Interpretação dos Gráficos
- **Regiões de Decisão:** Mostram como o Perceptron separa as classes
- **Convergência:** Demonstram se e quando o algoritmo para de aprender
- **Matrizes de Confusão:** Detalham tipos específicos de erros
- **Comparações:** Evidenciam o impacto de diferentes configurações

## Dependências

O projeto utiliza apenas bibliotecas padrão do ecossistema Python científico:
- **numpy:** Operações matemáticas e arrays
- **matplotlib:** Visualização de gráficos
- **scikit-learn:** Datasets e métricas de avaliação

## Troubleshooting

### Problemas Comuns

1. **Erro de encoding:** Se houver problemas com caracteres especiais, certifique-se de que seu terminal suporta UTF-8
2. **Matplotlib não mostra gráficos:** Instale um backend apropriado ou execute em ambiente com interface gráfica
3. **Scripts param quando solicitam paciência:** Digite um número inteiro (recomendado: 10)

### Suporte

Para análise detalhada dos resultados e metodologia, consulte o arquivo `RELATORIO.md`.