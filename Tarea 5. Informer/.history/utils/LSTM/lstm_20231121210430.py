
import numpy as np

class LSTM:
    
    def __init__(self, input_size, hidden_size):
         # Dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.Wi = 0 #Wi is the weight matrix for the input gate
        self.Wf = 0 #Wf is the weight matrix for the forget gate
        self.Wc = 0 #Wc is the weight matrix for the cell gate
        self.Wo = 0 #Wo is the weight matrix for the output gate
        
        #=============Matrix for the bias vector for each gate================
        self.bi = [] #bi is the bias vector for the input gate
        self.bf = [] #bf is the bias vector for the forget gate
        self.bc = [] #bc is the bias vector for the cell gate
        self.bo = [] #bo is the bias vector for the output gate
        
        
        # Initialize weights
        self._initialize_weights()
        
        
        
    def _initialize_weights(self):
        # Initialize weights for the input gate, forget gate, cell gate, and output gate
        # Plus biases for each gate
        self.Wi = np.random.randn(self.hidden_size, self.input_size + self.hidden_size)
        self.Wf = np.random.randn(self.hidden_size, self.input_size + self.hidden_size)
        self.Wc = np.random.randn(self.hidden_size, self.input_size + self.hidden_size)
        self.Wo = np.random.randn(self.hidden_size, self.input_size + self.hidden_size)

        self.bi = np.zeros(self.hidden_size)
        self.bf = np.zeros(self.hidden_size)
        self.bc = np.zeros(self.hidden_size)
        self.bo = np.zeros(self.hidden_size)
   
 
    def forward_pass(self, inputs):
        
        #Reselvamos memoria para las salidas 
        h_prev = np.zeros(self.hidden_size)  # Previous hidden state
        c_prev = np.zeros(self.hidden_size)  # Previous cell state

        for t in range(len(inputs)): # Por cada una de mis X
            combined = np.concatenate((h_prev, inputs[t])) # Concateno la entrada con la salida anterior

            # Input gate
            i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)

            # Forget gate
            f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)

            # ( Cell gate ) 
            # Aplicamos la tanh  al producto punto de la matriz de pesos y la concatenacion de la salida anterior con la entrada 
            # despues le sumamos el bias
            g = self.tanh(np.dot(self.Wc, combined) + self.bc) 

            # ( Output gate )
            # Aplicamos la sigmoid al producto punto de la matriz de pesos y la concatenacion de la salida anterior con la entrada
            # despues le sumamos el bias
            o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)

            #==== *Actualizacion ( Cell state )======
            # Actualizamos el estado de la celda
            c = f * c_prev + i * g

            # Hidden state
            # Aplicamos la tanh al estado de la celda y lo multiplicamos por la salida de la puerta de salida
            h = o * self.tanh(c)

            # Update states
            # Actualizamos el estado de la celda y la salida anterior
            # Esto dara como resultado la salida de la celda que vendria siendo la salida de la red o la entrada de la siguiente celda..
            
            h_prev = h # Actualizamos la salida anterior
            c_prev = c # Actualizamos el estado de la celda

        return h, c

    def backward_pass(self, dloss):
        
        def backward_pass(self, dH, inputs, H, C):
            # Inicializa los gradientes acumulativos para los pesos y biases
            dWi, dWf, dWc, dWo = [np.zeros_like(w) for w in [self.Wi, self.Wf, self.Wc, self.Wo]]
            dbi, dbf, dbc, dbo = [np.zeros_like(b) for b in [self.bi, self.bf, self.bc, self.bo]]

            # Inicializa los gradientes para los estados ocultos y de la celda
            dH_next = np.zeros_like(self.hidden_size)
            dC_next = np.zeros_like(self.hidden_size)

            # Bucle a través de los pasos de tiempo en orden inverso
            for t in reversed(range(len(inputs))):
                # Calcula gradientes para los estados ocultos y de la celda
                # Ten en cuenta que necesitarás derivar estas fórmulas basándote en la función de activación y la estructura LSTM
                # Estos son solo ejemplos genéricos y no representan cálculos precisos
                dH_total = dH[t] + dH_next
                dC_total = dC_next  (do * tanh_derivative(C[t]) * sigmoid_derivative(o[t]))

                # Calcular gradientes para los gates
                # Estos cálculos dependen de las operaciones realizadas en el forward pass
                di = dC_total * g[t] * sigmoid_derivative(i[t]) # Gradiente para el input gate
                df = dC_total * C_prev[t] * sigmoid_derivative(f[t]) # Gradiente para el forget gate
                do = dH_total * np.tanh(C[t]) * sigmoid_derivative(o[t])  # Gradiente para el output gate
                dg = dC_total * i[t] * tanh_derivative(g[t]) # Gradiente para el cell gate


                # Calcular gradientes para los pesos y biases
                dWi += np.dot(di, H_prev[t])
                dWf += np.dot(df, H_prev[t])
                dWc += np.dot(dg, H_prev[t])
                dWo += np.dot(do, H_prev[t])

                dbi += np.sum(di, axis=0)
                dbf += np.sum(df, axis=0)
                dbc += np.sum(dg, axis=0)
                dbo += np.sum(do, axis=0)

                # Preparar los gradientes para el siguiente paso de tiempo
                dH_next =  np.dot(self.Wi.T, di) + np.dot(self.Wf.T, df) +  # Gradiente para el siguiente estado oculto
                dC_next = dC_total * f[t] # Gradiente para el siguiente estado de la celda

            # Actualizar los pesos y biases
            # Nota: Necesitarás un método o una lógica adicional para la actualización real de los pesos
            self.Wi -= dWi
            self.Wf -= dWf
            self.Wc -= dWc
            self.Wo -= dWo

            self.bi -= dbi
            self.bf -= dbf
            self.bc -= dbc
            self.bo -= dbo

    def train(self, training_data, epochs):
        for epoch in range(epochs):
            for data in training_data:
                # Forward pass
                outputs = self.forward_pass(data['inputs'])
                
                # Compute loss
                loss = self.compute_loss(outputs, data['targets'])

                # Backward pass
                self.backward_pass(loss)

                # Update weights
                self.update_weights()

    def predict(self, input_data):
        outputs = self.forward_pass(input_data)
        return outputs[-1]  # Return last output as prediction


# Example usage
lstm = LSTM(input_size=..., hidden_size=...)
lstm.train(training_data, epochs=...)
predictions = lstm.predict(test_data)