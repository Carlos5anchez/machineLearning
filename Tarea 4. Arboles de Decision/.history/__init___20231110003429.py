

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



# decisionTreeClassifier = DecisionTreeClassifier(max_depth=130)
# decisionTreeClassifier.fit(X_train,y_train)
# predicted=decisionTreeClassifier.predict(X_test)
# print(predicted)
# print(y_test)
# metricas= Metricas(y_test,predicted)
# metricas.get_all()
errores=[]
rango=range(1,10)
for i in rango:
    decisionTreeClassifier = DecisionTreeClassifier(max_depth=i)
    decisionTreeClassifier.fit(X_train,y_train)
    predicted=decisionTreeClassifier.predict(X_test)
    # print(predicted)
    # print(y_test)
    metricas= Metricas(y_test,predicted)
    mse=metricas.mean_square()
    errores.append(mse)
    


# Gráfico de dispersión de los valores reales vs. predichos
plt.figure(figsize=(12, 6))
sns.scatterplot(x=rango, y=mse)
plt.xlabel('Max Depth')
plt.ylabel('MSE')
plt.title('Rendimento de Arbol de Decision')
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predicted)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()