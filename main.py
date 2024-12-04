import pandas as pd
from src.preprocesamiento import cargar_datos, preprocesar_datos
from src.clasificador import entrenar_modelo, evaluar_modelo

def main():
    # Cargar el dataset
    ruta = "data/iris.csv"  # Asegúrate de que el archivo iris.csv esté en la carpeta 'data'
    df = cargar_datos(ruta)
    
    # Preprocesar los datos
    X_train, X_test, y_train, y_test = preprocesar_datos(df)
    
    # Entrenar el modelo
    modelo = entrenar_modelo(X_train, y_train)
    
    # Evaluar el modelo
    evaluar_modelo(modelo, X_test, y_test)

if __name__ == "__main__":
    main()
