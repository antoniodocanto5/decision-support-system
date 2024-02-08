import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def loading_animation(text):
    time.sleep(1)
    print(text) 
    time.sleep(1)
    for _ in range(10):  
        print(".", end="", flush=True)  
        time.sleep(0.2)  
    print("\n")

    



data = pd.read_csv('netflix_data.csv')

#Verificar Valores Faltando
loading_animation("Verificando Valores Faltando")
print(data.isnull().sum())
time.sleep(2)
print("\n")

#Eliminar Colunas com Valores Faltando
loading_animation("Eliminando Colunas com Valores Faltando")
data.dropna(inplace=True)
time.sleep(2)
print("\n")

#Verificar o tipo de dados
loading_animation("Verificando o tipo dos Dados")
print(data.dtypes)
time.sleep(2)
print("\n")

#Verificar duplicados
loading_animation("Verificando os Duplicados")
print(data.duplicated().sum())
time.sleep(2)
print("\n")

#Eliminar duplicados
loading_animation("Eliminando os Duplicados")
data.drop_duplicates(inplace=True)
time.sleep(2)
print("\n")

#Verificar Outliers

#Normalizar

#Salvar dados