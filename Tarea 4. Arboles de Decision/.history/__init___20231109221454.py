

import pandas as pd
from sklearn.model_selection import train_test_split


#============Librerias propias================
from utils.arbol_decision import DecisionTreeClassifier


# Importamos el dataset necesario
data=pd.read_csv('./outputs/Salary_Dataset_with_Extra_Features.csv') #Cargamos el dataset

X=data.iloc[:,0:7].values #Cargamos las columnas de 0 a 7
y=data.iloc[:,8].values #Cargamos la columna 8

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


decisionTreeClassifier = DecisionTreeClassifier(max_depth=3)
decisionTreeClassifier.fit(X_train,y_train)
decisionTreeClassifier.predict(X_test)