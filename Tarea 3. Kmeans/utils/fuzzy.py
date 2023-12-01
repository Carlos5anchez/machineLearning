import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

class FuzzyCKMeans:
    # Constructor
    def __init__(self, df: pd.DataFrame, k: int, max_iter: int = 100, m: float = 2.0):
        self.data = df  # Dataframe de entrada
        self.k = k  # Numero de clusters
        self.max_iter = max_iter  # Numero maximo de iteraciones
        self.m = m  # Coeficiente de difusión
        self.centroids = None  # Centroides
        self.u = None  # Matriz de pertenencia

    def train(self):
        # Se normaliza los datos para que el algoritmo funcione más eficientemente
        data_norm = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        data_norm = data_norm.values.T  # La librería espera los datos en formato (variables, muestras)

        # Se aplica el algoritmo Fuzzy C-Means
        self.centroids, self.u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data_norm, self.k, self.m, error=0.005, maxiter=self.max_iter, init=None)

        # Se asignan los clusters a los datos originales
        cluster_assignment = np.argmax(self.u, axis=0)
        return cluster_assignment, self.centroids, self.u