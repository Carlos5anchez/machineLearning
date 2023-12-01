from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KNN:
    # Constructor
    def __init__(self, train_data: pd.DataFrame, train_labels: pd.Series, k: int = 3):
        self.train_data = train_data
        self.train_labels = train_labels
        self.k = k

    @staticmethod
    def _compute_distances(test_point, train_data):
        # Calcula las distancias euclidianas entre un punto de prueba y todos los puntos de entrenamiento
        distances = distance.cdist(test_point, train_data, 'euclidean').flatten()
        return distances

    def _get_k_nearest_neighbors_labels(self, distances):
        # Toma los k índices de las distancias más pequeñas y recupera sus etiquetas correspondientes
        k_indices = distances.argsort()[:self.k]
        k_nearest_labels = self.train_labels.iloc[k_indices].values
        return k_nearest_labels

    @staticmethod
    def _majority_vote(neighbors_labels):
        # Realiza una votación mayoritaria y devuelve la etiqueta más común entre los k vecinos más cercanos
        vote_counts = Counter(neighbors_labels)
        most_common_label = vote_counts.most_common(1)[0][0]
        return most_common_label

    def predict(self, test_data: pd.DataFrame):
        predictions = []
        for _, test_row in test_data.iterrows():
            test_point = test_row.values.reshape(1, -1)
            distances = self._compute_distances(test_point, self.train_data.values)
            neighbors_labels = self._get_k_nearest_neighbors_labels(distances)
            prediction = self._majority_vote(neighbors_labels)
            predictions.append(prediction)
        return np.array(predictions)
    def get_k_neighbors(self, test_point: np.array):
        distances = self._compute_distances(test_point.reshape(1, -1), self.train_data.values)
        
       
        
        k_indices = distances.argsort()[:self.k]
        
     
        
        neighbors = self.train_data.iloc[k_indices].values
        
     
        
        return neighbors
    def elbow_method(self, test_data, test_labels, max_k):
        errors = []
        
        for k in range(1, max_k + 1):
            self.k = k
            predictions = self.predict(test_data)
            error = 1 - accuracy_score(test_labels, predictions)
            errors.append(error)
            
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), errors, marker='o', linestyle='--')
        plt.xlabel('Número de Vecinos (k)')
        plt.ylabel('Tasa de Error')
        plt.title('Método del Codo para determinar valor de k')
        plt.show()

    def evauate(self,predictions, true_labels):
        # Calcula la precisión, recuperación y puntuación F1
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        f1 = f1_score(true_labels, predictions, average='macro')

        return accuracy, precision, recall, f1