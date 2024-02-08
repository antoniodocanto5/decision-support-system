import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# A análise de séries temporais é adequada para conjuntos de dados nos quais observações são coletadas
#em intervalos regulares ao longo do tempo. No seu caso, você possui dados diários do mercado de ações, 
#então a análise de séries temporais é altamente relevante.


# Load the dataset
data = pd.read_csv('netflix_data.csv')

# Convert 'date' column to datetime format and set it as the index

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the closing prices over time
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], color='blue')
plt.title('Netflix Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.grid(True)
plt.show()

# Perform seasonal decomposition
result = seasonal_decompose(data['close'], model='additive', period=30)  # Assuming seasonality period of 30 days
result.plot()
plt.show()