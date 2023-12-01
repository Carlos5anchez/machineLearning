import math


class Metricas:
    def __init__(self, real_values, predict_values):
      self.real_values = real_values
      self.predict_values = predict_values
    
    #Mean Squared Error (MSE)
    def mean_square(self):
        squared_errors = [(real - pred) ** 2 for real, pred in zip( self.real_values,  self.predict_values)]
        mse = sum(squared_errors) / len( self.real_values)
        print(f"Mean Squared Error: {mse}")
        return mse
    
    #Root Mean Squared Error (RMSE)
    def root_mean_square(self):
        squared_errors = [(real - pred) ** 2 for real, pred in zip( self.real_values,  self.predict_values)]
        mse = sum(squared_errors) / len( self.real_values)
        rmse = math.sqrt(mse)
        print(f"Root Mean Squared Error: {rmse}")
        return rmse
        
    # Mean Absolute Error (MAE)
    def mean_absolute(self):
        absolute_errors = [abs(real - pred) for real, pred in zip( self.real_values,  self.predict_values)]
        mae = sum(absolute_errors) / len( self.real_values)
        print(f"Mean Absolute Error: {mae}")
        return mae
        
    #Relative Square Error (RSE)
    def relative_square(self):
        squared_errors = [(real - pred) ** 2 for real, pred in zip( self.real_values,  self.predict_values)]
        average = sum(squared_errors) / len( self.real_values)
        rse = average / sum([(real - average) ** 2 for real in  self.real_values])
        print(f"Relative Square Error: {rse}")
        return rse    
    
    # Relative Absolute Error (RAE)
    def relative_absolute(self):
        absolute_errors = [abs(real - pred) for real, pred in zip( self.real_values,  self.predict_values)]
        average = sum(absolute_errors) / len( self.real_values)
        rae = average / sum([abs(real - average) for real in  self.real_values])
        print(f"Relative Absolute Error: {rae}")
    
    # Correlation Coeficient (CC)
    def correlation_coeficient(self):
        try:
          average_real = sum( self.real_values) / len( self.real_values)
          average_predict = sum( self.predict_values) / len( self.predict_values)
          numerator = sum([(real - average_real) * (pred - average_predict) for real, pred in zip( self.real_values,  self.predict_values)])
          denominator = math.sqrt(sum([(real - average_real) ** 2 for real in  self.real_values]) * sum([(pred - average_predict) ** 2 for pred in  self.predict_values]))
          cc = numerator / denominator
          print(f"Correlation Coeficient: {cc}")
        except:
          print('An exception occurred')

    # Coeficient of Determination (R²)
    def determination_coeficient(self):
        average_real = sum( self.real_values) / len( self.real_values)
        numerator = sum([(real - pred) ** 2 for real, pred in zip( self.real_values,  self.predict_values)])
        denominator = sum([(real - average_real) ** 2 for real in  self.real_values])
        r2 = 1 - (numerator / denominator)
        print(f"Determination Coeficient: {r2}")

    # Chi-Square (χ²)
    def chi_square(self):
        squared_errors = [(real - pred) ** 2 for real, pred in zip( self.real_values,  self.predict_values)]
        chi = sum(squared_errors) / sum( self.real_values)
        print(f"Chi-Square: {chi}")
    
    def get_all(self):
        self.mean_square()
        self.root_mean_square()
        self.mean_absolute()
        self.relative_square()
        self.relative_absolute()
        self.correlation_coeficient()
        self.determination_coeficient()
        self.chi_square()