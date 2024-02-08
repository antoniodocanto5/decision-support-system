import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Carregar o conjunto de dados
data = pd.read_csv('netflix_data.csv')

# Preparar os dados
X = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]  # Features
y = data['Close']  # Target variable (preço de fechamento)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo de árvore de decisão para regressão
model = DecisionTreeRegressor()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print("Erro quadrático médio (MSE):", mse)
