import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Realizar agrupamento k-means (uma técnica básica de mineração de dados)
from sklearn.cluster import KMeans

# Carregar o conjunto de dados
netflix = pd.read_csv('netflix_data.csv')

# Exibir as primeiras 5 linhas do conjunto de dados
print("Primeiras 5 linhas do conjunto de dados:")
print(netflix.head())

# Exibir estatísticas resumidas
print("\nEstatísticas resumidas:")
print(netflix.describe())

# Verificar se há valores faltantes
print("\nValores faltantes:")
print(netflix.isnull().sum())

# Limpar o conjunto de dados removendo todas as linhas com valores faltantes
netflix_limpo = netflix.dropna()

# Exibir o número de linhas antes e depois da limpeza
print("\nNúmero de linhas antes da limpeza:", len(netflix))
print("Número de linhas após a limpeza:", len(netflix_limpo))

# Visualizar a distribuição das características usando histogramas
netflix_limpo.hist()
plt.show()

# Verificar colunas com valores não numéricos
for col in netflix_limpo.columns:
    if netflix_limpo[col].dtype == 'object':
        try:
            pd.to_numeric(netflix_limpo[col])
        except ValueError:
            print(f"A coluna {col} contém valores não numéricos.")

# Remover a coluna "Date" do DataFrame
netflix_limpo_sem_data = netflix_limpo.drop(columns=['Date'])

# Calcular e visualizar a correlação entre as características restantes
plt.figure(figsize=(8, 6))
sns.heatmap(netflix_limpo_sem_data.corr(), annot=True, cmap="coolwarm")
plt.show()


# Exibir as primeiras 5 linhas do conjunto de dados com rótulos de cluster
print("\nPrimeiras 5 linhas do conjunto de dados com rótulos de cluster:")
print(netflix_limpo.head())