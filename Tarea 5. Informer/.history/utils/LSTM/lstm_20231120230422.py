
import numpy as np

class SimpleLSTM:
    
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
        
        #Reselvamos memoria para las salidas y los estados
        h_prev = np.zeros(self.hidden_size) 
        c_prev = np.zeros(self.hidden_size)

        for t in range(len(inputs)): # Por cada uno de mis entradas
            combined = np.concatenate((h_prev, inputs[t])) # Concateno la entrada con la salida anterior

            # Input gate
            i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)

            # Forget gate
            f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)

            # Cell gate
            g = self.tanh(np.dot(self.Wc, combined) + self.bc)

            # Output gate
            o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)

            # Cell state
            c = f * c_prev + i * g

            # Hidden state
            h = o * self.tanh(c)

            # Update states
            h_prev = h
            c_prev = c

        return h, c

    def backward_pass(self, dloss):
        pass
        # Implement the backward pass
        # ...

    def train(self, training_data, epochs):
        pass
        # Training method
        # ...

    def predict(self, input_data):
        pass
        # Prediction method
        # ...

# Example usage
lstm = SimpleLSTM(input_size=..., hidden_size=...)
lstm.train(training_data, epochs=...)
predictions = lstm.predict(test_data)