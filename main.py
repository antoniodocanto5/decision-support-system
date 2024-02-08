from telnetlib import SB
import pandas as pd

# Realizar agrupamento k-means (uma técnica básica de mineração de dados)
from sklearn.cluster import KMeans

# Carregar o conjunto de dados
netflix = pd.read_csv('netflix_data.csv')

# Exibir as primeiras 5 linhas do conjunto de dados
print("Primeiras 5 linhas do conjunto de dados:")
print(netflix.head(5))


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

