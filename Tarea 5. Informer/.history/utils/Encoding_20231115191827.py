import numpy as np

class PositionalEncoding:
    def __init__(self, max_seq_len, d_model):
        self.d_model = d_model # Dimensión del espacio de representación
        self.max_seq_len = max_seq_len # Longitud máxima de la secuencia de entrada
        self.positional_encoding = self._init_positional_encoding() # Matriz de codificación posicional

    def _init_positional_encoding(self):
        # Inicialización de la matriz de codificación posicional con ceros (f=Longitud de secuencia, c=Dimensión del espacio de representación )
        pos_encoding = np.zeros((self.max_seq_len, self.d_model))  
        for pos in range(self.max_seq_len): #Filas
            for i in range(0, self.d_model, 2): #Columnas
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i)/self.d_model))) #Columnas pares
                pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/self.d_model))) #Columnas impares
        return pos_encoding # Matriz de codificación posicional