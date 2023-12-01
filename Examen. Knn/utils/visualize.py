import matplotlib.pyplot as plt
import numpy as np

dark_palette = {
    0: 'black',   # Color para la etiqueta 0
    1: 'darkblue' # Color para la etiqueta 1
}


def displayKNN3DChart(knn_model, test_points, train_data,  labels):
    # Crear figura y eje 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    train_data = train_data.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    # Graficar los puntos de prueba
    ax.scatter(test_points['Pclass'], test_points['Age'], test_points['Fare'], c='red', marker='o', label='Datos de prueba')
    
    # Graficar los datos de entrenamiento
    for label in labels.unique():
        subset = train_data[labels == label]
        ax.scatter(subset['Pclass'], subset['Age'], subset['Fare'], c=dark_palette[label], label=f'Datos de entrenamiento {label}')
    
    # Calcular los k vecinos más cercanos para cada punto de prueba y dibujar líneas
    for _, test_row in test_points.iterrows():
        test_point = test_row[["Pclass",'Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']].values
        neighbors = knn_model.get_k_neighbors(test_point)
        for neighbor in neighbors:
            ax.plot([test_point[0], neighbor[0]], [test_point[1], neighbor[1]], [test_point[2], neighbor[2]], c='black')
    
    # Configurar etiquetas de los ejes y título
    ax.set_xlabel('Pclass')
    ax.set_ylabel('Age')
    ax.set_zlabel('Fare')
    ax.set_title('Visualización K-NN')
    ax.legend()
    plt.show()

def plot_metrics(accuracy, precision, recall, f1):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'cyan'])
    plt.xlabel('Score')
    plt.ylabel('Metric')
    plt.xlim(0, 1)
    plt.title('Evaluation Metrics')
    for index, value in enumerate(values):
        plt.text(value, index, f'{value:.4f}')
    plt.tight_layout()
    plt.show()