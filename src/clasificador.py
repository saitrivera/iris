from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def entrenar_modelo(X_train, y_train):
    """Entrena el modelo usando Random Forest"""
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    """Evalúa el rendimiento del modelo"""
    # Predicciones sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)
    
    # Precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.2f}")
    
    # Reporte de clasificación
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Verdaderos")
    plt.show()
