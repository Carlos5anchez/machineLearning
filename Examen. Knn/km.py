

import pandas as pd
from sklearn.model_selection import train_test_split


#============Librerias propias================
from utils.Knn import KNN

from utils.visualize import displayKNN3DChart


# Importamos el dataset necesario
data=pd.read_csv('./outputs/BD Titanic Completa.csv') #Cargamos el dataset

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Verificar los tamaños de los conjuntos de entrenamiento y prueba
print(f"Tamaño del conjunto de entrenamiento: {len(train_df)}")
print(f"Tamaño del conjunto de prueba: {len(test_df)}")

trainData=train_df[["Pclass",'Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]

knn=KNN(train_data=trainData,train_labels=train_df['Survived'],k=4) #Creamos el objeto KNN

predictions = knn.predict(train_df[["Pclass",'Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']]) #Entrenamos el modelo
eval=knn.evauate(predictions,train_df['Survived']) #Evaluamos el modelo
print({
    "accuracy":eval[0],
    "precision":eval[1],
    "recall":eval[2],
    "f1":eval[3]
})

pd.DataFrame(predictions).to_csv('./outputs/predictions.csv',index=False) #Guardamos las predicciones


# Mostrar visualización 3D
displayKNN3DChart(knn,train_df[["Pclass",'Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']], test_df[["Pclass",'Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']], train_df['Survived'])
knn.elbow_method(trainData,train_df['Survived'],20)