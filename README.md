# Aprendizagem Computacional 25/26

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellowgreen.svg)
![Status](https://img.shields.io/badge/Status-Completo-success.svg)

Repositório de trabalhos práticos da unidade curricular **Aprendizagem Computacional** do curso de **Licenciatura em Ciências da Computação** na Universidade do Minho.

---

## 📑 Índice

1. [📚 Estrutura do Repositório](#-estrutura-do-repositório)
2. [� Aulas Teóricas](#-aulas-teóricas)
3. [📂 TP1 - Python Básico](#-tp1---python-básico)
4. [📊 TP2 - Machine Learning](#-tp2---machine-learning-k-means--pca)
5. [🎓 Mini-Exercises](#-mini-exercises-tp2)
6. [🛠️ Tecnologias Utilizadas](#️-tecnologias-utilizadas)
7. [🚀 Como Executar](#-como-executar)
8. [📖 Recursos Úteis](#-recursos-úteis)

---

## 📚 Estrutura do Repositório

```
AC26/
├── Teoricas/
│   └── PP_T_aula01.ipynb    # Data Mining e CRISP-DM
│
├── TP1/
│   ├── Exercises.ipynb      # Exercícios de Python básico
│   ├── sample.json          # Dados de exemplo (JSON)
│   └── temperatures.txt     # Dataset de temperaturas
│
└── TP2/
    └── KMeans_PCA.ipynb     # Machine Learning: Clustering e PCA
```

---

## � Aulas Teóricas

Material das aulas teóricas com conceitos fundamentais de Data Mining e Machine Learning.

### 📝 Aula 01 - Data Mining e CRISP-DM

**Ficheiro:** `PP_T_aula01.ipynb`

#### 🔄 CRISP-DM (Cross-Industry Standard Process for Data Mining)
Metodologia standard para projetos de Data Mining:

1. **Business Understanding** - Objetivos de negócio e planeamento do projeto
2. **Data Understanding** - Coleta, descrição, visualização e verificação de qualidade
3. **Data Preparation** - Seleção, limpeza, integração, formatação e transformação
4. **Modeling** - Seleção de técnicas, design de testes e construção de modelos
5. **Evaluation** - Avaliação de resultados e revisão do processo
6. **Deployment** - Implementação e decisões sobre próximos passos

#### 📊 Tarefas Comuns de Data Mining

**Preditivas (Predictive):**
- **Classification** - Prever a classe de um objeto
- **Regression** - Prever um valor numérico
- **Time Series Analysis** - Prever próximo valor numa série temporal

**Descritivas (Descriptive):**
- **Clustering** - Agrupar dados por similaridade
- **Summarization** - Resumir e generalizar dados
- **Association Rules** - Descobrir conexões entre itens
- **Sequence Discovery** - Encontrar padrões em dados sequenciais

#### 🧹 Data Cleaning - Passos Fundamentais
1. Importar dados
2. Juntar datasets (merge)
3. Tratar dados em falta (missing data)
4. Estandardização
5. Normalização

#### ⚠️ Desafios Comuns
- Qualidade de dados pode ser pobre
- Dados podem estar em falta ou com ruído
- Padrões encontrados podem não ser exatos, úteis ou interessantes

---

## �📂 TP1 - Python Básico

Exercícios de introdução à programação em Python divididos em três níveis de dificuldade.

### 🟢 Python I - Fundamentos Básicos

1. **Age Checker** - Classificação por faixas etárias (`if`, `elif`, `else`)
2. **String Cleaner** - Manipulação de strings (`.strip()`, `.lower()`, `.replace()`)
3. **CompScience FizzBuzz** - Lógica condicional com múltiplos de 3 e 5
4. **Guess the Number** - Jogo de adivinhação com loops
5. **Factorial** - Cálculo iterativo de fatorial

### 🟡 Python II - Estruturas de Dados

1. **Palindrome** - Verificação de palavras palindrómicas
2. **Normalize** - Normalização de vetores usando NumPy
3. **Frequency Table** - Contagem de elementos em listas
4. **List/Dict Comprehension** - Comprehensions avançadas

### 🔴 Python III - Ficheiros e OOP

1. **Input/Output** - Leitura e processamento de `temperatures.txt`
2. **JSON** - Parsing e manipulação de dados JSON (`sample.json`)
3. **Try/Except** - Tratamento de erros e exceções
4. **Recursion** - Implementação de funções recursivas
5. **Classes** - Sistema de gestão de biblioteca (OOP)

### 📁 Ficheiros de Dados (TP1)

- `temperatures.txt` - Dataset de temperaturas para exercícios de I/O
- `sample.json` - Ficheiro JSON para exercícios de parsing

---

## 📊 TP2 - Machine Learning: K-Means & PCA

Trabalho prático sobre aprendizagem não supervisionada com clustering e redução de dimensionalidade.

### 📚 Estrutura do Notebook

#### **0) Setup and Data Loading**

- Importação de bibliotecas (NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn)
- Configuração do ambiente de trabalho

#### **1) What is Machine Learning?**

- Introdução à Aprendizagem Computacional
- Tipos de Machine Learning: Supervised vs Unsupervised

#### **2) Unsupervised Learning Overview**

- Conceitos fundamentais de aprendizagem não supervisionada
- Clustering e Dimensionality Reduction

#### **3) Data Preparation and Exploration**

- Dataset: **Chemical Compounds** (peso molecular, solubilidade, ponto de fusão)
- Análise exploratória de dados (EDA)
- Data Preprocessing e normalização com `StandardScaler`
- **Mini-exercises**: Matriz de correlação entre variáveis

#### **4) Clustering Analysis**

- **4.1 K-Means Clustering**
  - Algoritmo K-Means step-by-step
  - Métricas: Inertia, Silhouette Score
  - Elbow Method para determinar k ótimo
  - Visualização de clusters
- **Mini-exercises**: Comparação com Agglomerative Clustering

#### **5) Dimensionality Reduction with PCA**

- **Principal Component Analysis (PCA)**
  - Redução de dimensionalidade
  - Variância explicada e cumulativa
  - Scree Plot e visualização de componentes principais
- **PCA on Chemical Dataset**
  - Aplicação prática do PCA
  - Transformação de novos compostos
- **Mini-exercises**:
  1. Quantos PCs são necessários para capturar 95% da variância?
  2. Transformar novo composto usando PCA ajustado

#### **6) Real-World Example: Palmer Penguins**

- Dataset: **Penguins** (Adelie, Chinstrap, Gentoo)
- Features: bill_length, bill_depth, flipper_length, body_mass
- Clustering e comparação com labels verdadeiros
- **Confusion Matrix** e **Adjusted Rand Index (ARI)**

### 🎯 Datasets Utilizados

| Dataset                  | Features                                                      | Classes                   | Objetivo                      |
| ------------------------ | ------------------------------------------------------------- | ------------------------- | ----------------------------- |
| **Chemical Compounds**   | molecular_weight, log_solubility, melting_point               | -                         | Clustering não supervisionado |
| **Palmer Penguins**      | bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g | Adelie, Chinstrap, Gentoo | Validação de clustering       |
| **Wine** (Mini-exercise) | 13 features químicas                                          | 3 cultivares              | Aplicação prática             |

### 🔑 Conceitos-Chave Aprendidos

- **Clustering**: K-Means, Agglomerative/Hierarchical Clustering
- **Redução de Dimensionalidade**: PCA, Explained Variance
- **Métricas**: Silhouette Score, Inertia, Adjusted Rand Index (ARI)
- **Preprocessing**: StandardScaler, feature scaling
- **Visualização**: Scatter plots, heatmaps, confusion matrices

---

## 🛠️ Tecnologias Utilizadas

### TP1 - Python Básico

| Biblioteca                  | Descrição           | Uso                          |
| --------------------------- | ------------------- | ---------------------------- |
| **NumPy**                   | Computação numérica | Normalização de vetores      |
| **Python Standard Library** | Funções built-in    | I/O, JSON, classes, recursão |

### TP2 - Machine Learning

| Biblioteca       | Versão | Descrição                                                |
| ---------------- | ------ | -------------------------------------------------------- |
| **NumPy**        | Latest | Computação numérica e arrays multidimensionais           |
| **Pandas**       | Latest | Manipulação e análise de dados tabulares                 |
| **Matplotlib**   | Latest | Visualização de dados estática                           |
| **Seaborn**      | Latest | Visualização estatística avançada                        |
| **Scikit-Learn** | Latest | Machine Learning (KMeans, PCA, StandardScaler, métricas) |

---

## 🚀 Como Executar

### Requisitos:

- Python 3.8+
- Jupyter Notebook / VS Code com extensão Jupyter

### Instalação de Dependências:

**Para TP1 (Python Básico):**

```bash
pip install numpy
```

**Para TP2 (Machine Learning):**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**Ou instalar tudo de uma vez:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Executar Notebooks:

**Opção 1 - Jupyter Notebook:**

```bash
cd "c:\Users\Lenovo\OneDrive - Universidade do Minho\Documentos\Escolinha\AC26"
jupyter notebook
```

**Opção 2 - VS Code:**

- Abrir a pasta `AC26` no VS Code
- Instalar extensão "Jupyter" da Microsoft
- Abrir o ficheiro `.ipynb` desejado
- Selecionar kernel Python
- Executar células com `Shift+Enter`

---

## 📖 Recursos Úteis

### Documentação Oficial:

- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/pyplot.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial/introduction.html)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)

### Conceitos de Machine Learning:

- [K-Means Clustering Explained](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [PCA - Principal Component Analysis](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Silhouette Score](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)

---

## 🎓 Mini-Exercises (TP2)

O notebook TP2 contém três conjuntos de mini-exercises práticos:

1. **Data Exploration (~10 min)**
   - Criar matriz de correlação entre as três variáveis do `compounds_df`
   - Visualizar com heatmap

2. **Hierarchical Clustering (~15 min)**
   - Aplicar AgglomerativeClustering ao `features_scaled`
   - Comparar Silhouette Score com K-Means
   - Visualizar diferenças entre os métodos

3. **PCA (~10 min)**
   - Determinar quantos PCs são necessários para 95% da variância
   - Criar novo composto e transformá-lo com PCA ajustado
   - Dataset extra: **Wine** (clustering e PCA)

---

## 👤 Autor

**Universidade do Minho**  
Licenciatura em Ciências da Computação  
Aprendizagem Computacional 2025/2026

---

## 📝 Notas

- 📘 Os notebooks contêm explicações detalhadas e comentários em cada secção
- 📦 Os datasets estão incluídos nos respetivos diretórios (`TP1/`) ou carregados via `sklearn.datasets` (TP2)
- ⚡ Recomenda-se executar as células sequencialmente para evitar erros de dependências
- 🧪 Os mini-exercises são práticos e devem ser implementados durante o estudo
- 🔧 Certifique-se de que todas as dependências estão instaladas antes de executar os notebooks

---

**Última atualização:** Fevereiro 2026
