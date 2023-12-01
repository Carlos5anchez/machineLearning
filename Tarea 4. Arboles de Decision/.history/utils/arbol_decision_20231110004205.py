import numpy as np

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, gini=None, value=None):
        # ============== Para nodos de decisión ========================================
        # Un nodo de decision es un camino que se puede tomar en el que se evalua una condicion
        # Por ejemplo, si la edad es menor a 10, entonces se va por la izquierda, si no, por la derecha
        #==============================================================================
        # En este caso, feature_index es el indice de la columna que se va a evaluar
        # threshold es el valor que se va a evaluar
        # left es el camino que se va a tomar si se cumple la condicion
        # right es el camino que se va a tomar si no se cumple la condicion
        # gini es el indice de gini del nodo
        # Para hojas value es el valor que se va a predecir
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gini = gini
        # Para hojas
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y) # Crear el árbol de decisión, empezando por la raíz (Nota: no ponemos el parámetro depth debido a que es la raíz)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)] # Muestra el numero de clases
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionNode(value=predicted_class) # Creamos el nodo de decisión

        # Detenemos la división si se alcanza la profundidad máxima 
        if depth >= self.max_depth:
            return node

        if len(np.unique(y)) == 1: # Detenemos la división si solo hay una clase
            return node

        best_feature, best_threshold, best_gini = self._best_split(X, y)
        if best_gini is None:
            return node

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return DecisionNode(best_feature, best_threshold, left, right, best_gini)

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        m, n = X.shape
        best_gini = 1.0
        best_feature, best_threshold = None, None

        for feature_index in range(n):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = X[:, feature_index] >= threshold
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gini_left = self._gini(y[left_indices])
                gini_right = self._gini(y[right_indices])
                gini = (len(left_indices) * gini_left + len(right_indices) * gini_right) / m

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def predict(self, X):
        predictions = []
        for inputs in X:
            node = self.root
            while node.value is None:
                if inputs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return predictions


