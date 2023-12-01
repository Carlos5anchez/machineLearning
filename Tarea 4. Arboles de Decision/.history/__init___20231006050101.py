

import pandas as pd


#============Librerias propias================
from utils.kmeans import Kmeans
from utils.fuzzy import FuzzyCKMeans
from utils.visualize import display3DChart


# Importamos el dataset necesario
data=pd.read_csv('./outputs/Salary_Dataset_with_Extra_Features.csv') #Cargamos el dataset


#======== Aplica K-Means=============
#Instanciamos la clase Kmeans con nuestro dataset y el numero de clusters
numClusters=8 
#kmn=Kmeans(data, k=numClusters, max_iter=100) #Clase creada en utils/kmeans.py
#clusters, centroids,distancia = kmn.train() #Entrenamos el modelo

#======== Aplica Fuzzy C-Means=============
modelo_fuzzy = FuzzyCKMeans(data, k=numClusters)
clusters, centroids, matriz_u = modelo_fuzzy.train()
print(matriz_u)
# Seleccionar tres columnas para representar en 3D
dim_x = 2  # Índice de la primera dimensión (columna)
dim_y = 4  # Índice de la segunda dimensión (columna)
dim_z = 6  # Índice de la tercera dimensión (columna)

# # Define el rango de valores de k que quieres probar
# k_values = range(1, 40)

# kmn.codo(k_values)    
data['Cluster'] = clusters  # Añade las asignaciones al DataFrame

# Agrupa por cluster y calcula la media
cluster_means = data.groupby('Cluster').mean()

# Resetear índices para obtener un DataFrame limpio
cluster_means.reset_index(inplace=True)

# Visualizar o guardar los resultados
print(cluster_means["Salary"],["Cluster"])
        
display3DChart(clusters=clusters,data=data,centroids=centroids,numClusters=numClusters,indx_x=dim_x,indx_y=dim_y,indx_z=dim_z) #Clase creada en utils/visualize.py

