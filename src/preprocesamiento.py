# src/preprocesamiento.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_datos(ruta):
    """Carga el dataset desde un archivo CSV separado por tabulaciones"""
    df = pd.read_csv(ruta, sep='\t')  # Usar tabulador como delimitador
    print("Primeras filas del DataFrame:")
    print(df.head())  # Muestra las primeras filas
    print("Columnas disponibles:")
    print(df.columns)  # Muestra las columnas del DataFrame
    return df


def preprocesar_datos(df):
    """Preprocesa los datos para el entrenamiento"""
    # Limpiar posibles espacios en blanco en los nombres de las columnas
    df.columns = df.columns.str.strip()
    
    if 'species' not in df.columns:
        raise KeyError(f"La columna 'species' no existe en el DataFrame. Columnas disponibles: {df.columns}")
    
    # Separamos las características y la variable objetivo
    X = df.drop('species', axis=1)  # Variables predictoras
    y = df['species']               # Variable objetivo
    
    # Normalizamos las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividimos los datos en entrenamiento y prueba (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
