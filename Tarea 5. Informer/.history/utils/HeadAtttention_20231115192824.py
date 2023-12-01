import numpy as np



class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        #==========Inputs================
        self.d_model = d_model # Dimensión del espacio de representación
        self.num_heads = num_heads # Número de cabezas de atención   
        #============================
        
        self.depth = d_model // num_heads # Profundidad de cada cabeza de atención (d_model/num_heads)
      
        #====== Matrices de peso para las consultas, claves y valores======
        self.wq = np.random.randn(d_model, d_model)  #Matriz de peso para las consultas
        self.wk = np.random.randn(d_model, d_model) #Matriz de peso para las claves
        self.wv = np.random.randn(d_model, d_model)#Matriz de peso para los valores
     
     
        
    # Implementación del mecanismo de atención escalada por producto punto
    def scaled_dot_product_attention(self, queries, keys, values):
        
        #(swapaxes(-1, -2) es para transponer la matriz de claves
        matmul_qk = np.matmul(queries, keys.swapaxes(-1, -2)) / np.sqrt(self.depth) # Producto punto entre las consultas y las claves 
        attention_weights = np.softmax(matmul_qk, axis=-1) # Softmax sobre las filas de la matriz de producto punto
        output = np.matmul(attention_weights, values)
        return output

    def call(self, queries, keys, values):
        # Implementación de la atención de múltiples cabezas
        pass