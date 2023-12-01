

import pandas as pd


#============Librerias propias================
from utils.arbol_decision import DecisionTreeClassifier


# Importamos el dataset necesario
data=pd.read_csv('./outputs/Salary_Dataset_with_Extra_Features.csv') #Cargamos el dataset

decisionTreeClassifier = DecisionTreeClassifier(max_depth=3)
decisionTreeClassifier.fit(data.iloc[:, 0:7].values, data.iloc[:, 8].values)