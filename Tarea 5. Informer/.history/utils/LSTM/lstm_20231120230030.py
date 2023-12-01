import nltk
 

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
   
    def sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def tanh(self, x):
        return np.exp(x) - np.exp(-x) / (np.exp(x) + np.exp(-x))
     
    def forward_pass(self, inputs):
        pass
        # Implement the forward pass
        # ...

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