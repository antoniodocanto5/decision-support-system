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

# Visualizar a correlação entre as características usando um mapa de calor
# plt.figure(figsize=(8, 6))
# sns.heatmap(netflix_limpo.corr(), annot=True, cmap="coolwarm")
# plt.show()



# netflix_numerico = netflix_limpo[['comprimento_da_sépala', 'largura_da_sépala', 'comprimento_do_pétalo', 'largura_do_pétalo']]
# kmeans = KMeans(n_clusters=3, random_state=42).fit(netflix_numerico)

# Adicionar rótulos de cluster ao conjunto de dados
netflix_limpo['cluster'] = KMeans.labels_

# Exibir as primeiras 5 linhas do conjunto de dados com rótulos de cluster
print("\nPrimeiras 5 linhas do conjunto de dados com rótulos de cluster:")
print(netflix_limpo.head())