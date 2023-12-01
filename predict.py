import torch
from exp.exp_informer import Exp_Informer
from config import getConfig
import pandas as pd

def load_and_predict(args, model_path, prediction_data):
    # Asumiendo que 'args' es un objeto que contiene los argumentos necesarios para el modelo
    # 'model_path' es la ruta al modelo guardado
    # 'prediction_data' son los datos sobre los cuales quieres hacer la predicción

    # Configurar el uso de GPU si está disponible
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    # Crear una instancia de Exp_Informer
    exp = Exp_Informer(args)

    # Cargar el modelo
    #exp.model.load_state_dict(torch.load(model_path, map_location=args.device))

    # Realizar la predicción
    prediction = exp.predict(model_path,True)

    return prediction

# Configuración de los argumentos
args = getConfig()
# Añadir más configuraciones a args si es necesario

# Ruta al modelo guardado
model_path = 'Mejor'

# Datos para hacer la predicción
# Importamos las librerias necesarias
df=pd.read_csv('./data/ETT/DMN_Report_29-All.csv') #Cargamos el dataset
input_data = df[-args.seq_len:] 
# Llamada a la función
predicted_values = load_and_predict(args, model_path, input_data)

