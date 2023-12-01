import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Kmeans:
    # Contructor
    def __init__(self, df: pd.DataFrame, k: int, max_iter: int = 100):
      self.data= df #Dataframe de entrada
      self.k = k # Numero de clusters
      self.max_iter = max_iter # Numero maximo de iteraciones
      self.clusters = [] # Lista de clusters
      self.centroides = [] # Lista de centroides
    
    @staticmethod
    def _initialize_centroids(data, k):
      # Inicializa los centroides seleccionando aleatoriamente k puntos de datos
      indices = np.random.choice(len(data), k, replace=False) #Selecciona k indices aleatorios
      centroids = data.iloc[indices].values 
      return centroids
    
    @staticmethod
    def _assign_clusters(data, centroids):
      # Asigna cada punto de datos al centroide más cercano
      # Calcula la distancia de cada punto a cada centroide con norma euclidea
      distances = np.linalg.norm(data.values[:, np.newaxis, :] - centroids, axis=2) 
      clusters = np.argmin(distances, axis=1) #Sacamos indice del minimo de cada row
      return clusters
   
    @staticmethod
    def _update_centroids(data, clusters, k):
        # Actualiza los centroides calculando la media de los puntos asignados a cada cluster
        centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)]) #Cuando el indice de cada row que corresponde a la posicion del mas cercano, sea igual a i, se calcula la media de los puntos asignados a cada cluster
        return centroids

    @staticmethod
    def _distancia_InterCluster(data,k,clusters,centroids):
         # Calcula la SSE Sum of Squared Errors
      sse = 0
      for i in range(k):
          cluster_points = data[clusters == i]
          distance_to_centroid = np.linalg.norm(cluster_points - centroids[i], axis=1)
          sse += np.sum(np.square(distance_to_centroid))
          
      return sse
    
    def codo(self, k_range):
        sse_list = []
        for k in k_range:
            # Instancia y entrena el modelo Kmeans con k clusters
            kmeans = Kmeans(self.data, k=k, max_iter=100)
            _, _, sse = kmeans.train()
            sse_list.append(sse)
            
        # Grafica los resultados
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, sse_list, '-o')
        plt.title('Método del codo para optimos k')
        plt.xlabel('Numero de Clusters (k)')
        plt.ylabel('Suma de errores cuadrados (SSE)')
        plt.xticks(k_range)
        plt.show()
        plt.pause(100000)    
        return sse_list


       
      
    
    def train(self):
       centroids = self._initialize_centroids(self.data, self.k)

       for _ in range(self.max_iter):
          # Indices de los clusters
          clusters = self._assign_clusters(self.data, centroids)

          # Actualiza centroides
          new_centroids = self._update_centroids(self.data, clusters, self.k)

          # Verifica convergencia
          if np.allclose(centroids, new_centroids):
              break

          centroids = new_centroids
       
       distancia=self._distancia_InterCluster(self.data,self.k,clusters,centroids)
       return clusters, centroids,distancia