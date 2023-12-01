

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

#============Librerias propias================
from utils.arbol_decision import DecisionTreeClassifier
from utils.metricas import Metricas

# Importamos el dataset necesario
data=pd.read_csv('./outputs/diabetes.csv') #Cargamos el dataset

X=data.iloc[:,0:7].values #Cargamos las columnas de 0 a 7
y=data.iloc[:,8].values #Cargamos la columna 8

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=143)



decisionTreeClassifier = DecisionTreeClassifier(max_depth=130)
decisionTreeClassifier.fit(X_train,y_train)
predicted=decisionTreeClassifier.predict(X_test)
print(predicted)
print(y_test)
metricas= Metricas(y_test,predicted)
metricas.get_all()


# Gráfico de dispersión de los valores reales vs. predichos
plt.figure(figsize=(12, 6))
# sns.scatterplot(x=real_values, y=predict_values, alpha=0.6)
sns.barplot(x=y_test, y=predicted)  # Línea de identidad (perfecta predicción)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Valores Reales vs. Valores Predichos')
plt.grid(True)
plt.show()