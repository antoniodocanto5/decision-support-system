import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('netflix_data.csv')

# Convert 'date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the trading volume over time
plt.figure(figsize=(10, 6))
plt.plot(data['Volume'], color='green')
plt.title('Netflix Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.show()
