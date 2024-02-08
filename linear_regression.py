import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregar o conjunto de dados (substitua 'data.csv' pelo nome do seu arquivo CSV)
data = pd.read_csv('data.csv')

# Separar as variáveis independentes (X) e dependentes (y)
X = data[['feature1', 'feature2', 'feature3']]  # Substitua 'feature1', 'feature2', 'feature3' pelos nomes das suas características
y = data['target']  # Substitua 'target' pelo nome da sua variável alvo

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo usando o conjunto de treinamento
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE (Mean Squared Error): {mse}')
print(f'R2 Score: {r2}')

# Visualizar os resultados
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Valores Reais vs. Valores Previstos")
plt.show()
