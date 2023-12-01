import numpy as np

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, gini=None, value=None):
        # ============== Para nodos de decisión ========================================
        # Un nodo de decision es un camino que se puede tomar en el que se evalua una condicion
        # Por ejemplo, si la edad es menor a 10, entonces se va por la izquierda, si no, por la derecha
        #==============================================================================
        # En este caso: 
        # feature_index es el indice de la columna que se va a evaluar
        # threshold es el valor que se va a evaluar
        # left es el camino que se va a tomar si se cumple la condicion
        # right es el camino que se va a tomar si no se cumple la condicion
        # indice de gini del nodo
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
        self.root = self._crecer_arbol(X, y) # Crear el árbol de decisión, empezando por la raíz (Nota: no ponemos el parámetro depth debido a que es la raíz)

    def _crecer_arbol(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)] # Muestra el numero de clases
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionNode(value=predicted_class) # Creamos el nodo de decisión Raíz

        # Detenemos la división si se alcanza la profundidad máxima 
        if depth >= self.max_depth:
            return node

        if len(np.unique(y)) == 1: # Detenemos la división si solo hay una clase
            return node

        best_feature, best_threshold, best_gini = self._dividir_arbol(X, y) # Obtenemos el mejor split
        if best_gini is None:
            return node

        left_indices = X[:, best_feature] < best_threshold # Obtenemos los indices de la izquierda
        right_indices = X[:, best_feature] >= best_threshold # Obtenemos los indices de la derecha
        
        left = self._crecer_arbol(X[left_indices], y[left_indices], depth + 1) # Creamos el nodo de decisión izquierdo
        right = self._crecer_arbol(X[right_indices], y[right_indices], depth + 1) # Creamos el nodo de decisión derecho
        return DecisionNode(best_feature, best_threshold, left, right, best_gini) # Retornamos el nodo de decisión

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _dividir_arbol(self, X, y):
        numFilas, numCol = X.shape
        best_gini = 1.0 # Inicializamos la variable
        best_feature, best_threshold = None, None # Inicializamos las variables

        for dataColumn in range(numCol): # Recorremos las columnas
            thresholds = np.unique(X[:, dataColumn]) # Obtenemos los valores únicos de la columna, Son todas mis posibles condiciones
            for threshold in thresholds: # Recorremos los valores únicos
              
              # Por cada valor único, separamos los datos en dos grupos, los que cumplen la condición y los que no
                left_indices = X[:, dataColumn] < threshold # Obtenemos los indices de la izquierda (NO)
                right_indices = X[:, dataColumn] >= threshold # Obtenemos los indices de la derecha (SI)
                if len(left_indices) == 0 or len(right_indices) == 0: # Si no hay indices, 
                    continue 

                gini_left = self._gini(y[left_indices]) # Obtenemos el gini de la izquierda
                gini_right = self._gini(y[right_indices]) # Obtenemos el gini de la derecha
                gini = (len(left_indices) * gini_left + len(right_indices) * gini_right) / numFilas # Obtenemos el gini total

                if gini < best_gini: # Si el gini es menor vs el mejor gini anterior
                    best_gini = gini # Actualizamos el mejor gini
                    best_feature = dataColumn # Actualizamos el mejor feature
                    best_threshold = threshold # Actualizamos el mejor threshold

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


