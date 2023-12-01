import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Crear un DataFrame con las estadísticas
def getNumericStats(numeric_dataset: pd.DataFrame):
    
    # Calcular la media, suma de todos los valores entre la cantidad de valores
    media = numeric_dataset.sum() / numeric_dataset.count()

    # Calcular la moda (el valor más común)
    # Utilizamos apply para aplicar una función a cada columna, lambda para definir una función anónima 
    # value_counts para contar los valores y ordenarlos de mayor a menor,
    # despues obtenemos el primer valor (el más común)
    moda = numeric_dataset.apply(lambda x: x.value_counts().index[0]) 

    # Calcular el valor mínimo
    min = numeric_dataset.min()

    # Calcular el valor máximo
    max = numeric_dataset.max()



    # Calcular la desviación estándar
    # Utilizamos apply para aplicar una función a cada columna, lambda para definir una función anónima
    # x representa cada columna, (x - media)**2/n-1 es la diferencia entre cada valor y la media al cuadrado
    # sum() es la suma de todos los valores, numeric_dataset.count() es la cantidad de valores
    # al final dividimos la suma entre la cantidad de valores menos 1 
    varianza_muestral = ((numeric_dataset - media) ** 2).sum() / (numeric_dataset.count() - 1)

    # Calcula la desviación estándar (raíz cuadrada de la varianza muestral)
    desviacion_estandar = varianza_muestral.apply(lambda x: x ** 0.5)

    # Calcular la cantidad de datos faltantes por atributo
    datos_faltantes = numeric_dataset.isnull().sum()

    # Crear un DataFrame con las estadísticas
    estadisticas = pd.DataFrame({
        'Media': media,
        'Moda': moda,
        'Mínimo': min,
        'Máximo': max,
        'Desviación Estándar': desviacion_estandar,
        'Datos Faltantes': datos_faltantes
    })

    # Imprimir las estadísticas
    return estadisticas

# Visualizar las distribuciones de los datos
def get_numpy_distribucions(numeric_dataset: pd.DataFrame):   
    sns.set(style="whitegrid")#STYLE
    for columna in numeric_dataset.columns:
        print(columna)
        bins=len(columna) #Cantidad de barras en el histograma
        plt.figure(figsize=(10, 6))
        # Histograma
        sns.histplot(data=numeric_dataset[columna], bins=bins, kde=True) 
        plt.title(f'Distribución de {columna}')
        plt.ylabel('Densidad')
        plt.xlabel(columna)
        plt.xlim(numeric_dataset[columna].min(),numeric_dataset[columna].max() )
        if (columna=='Salary') :
            # Evitar notación científica en el eje x
            plt.ticklabel_format(style='plain', axis='x')
        plt.show()


# Visualizar las distribuciones de los datos en un bar chart   
def get_numpy_barChart(dataset: pd.DataFrame):   
    
    for columna in dataset.columns:
        conteo_categorias = dataset[columna].value_counts()
        plt.figure(figsize=(10, 6))

        # Histograma
        sns.barplot(x=conteo_categorias.index, y=conteo_categorias.values, palette="viridis") 
        plt.title(f'Distribución de {columna}')
        plt.ylabel('Densidad')
        plt.xlabel(columna)
        plt.xticks(rotation=45)  # Girar las etiquetas del eje x para una mejor legibilidad
        plt.show()