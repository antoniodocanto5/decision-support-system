from main import *
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# # Selecionar apenas as colunas numéricas para o agrupamento
# netflix_numerico = netflix_limpo_sem_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

# # Aplicar o algoritmo KMeans para realizar o agrupamento
# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans.fit(netflix_numerico)

# # Adicionar rótulos de cluster ao DataFrame
# netflix_limpo_sem_data['Cluster'] = kmeans.labels_

# # Exibir as primeiras 5 linhas do conjunto de dados com rótulos de cluster
# print("\nPrimeiras 5 linhas do conjunto de dados com rótulos de cluster:")
# print(netflix_limpo.head())