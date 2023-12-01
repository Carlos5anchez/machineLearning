from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize(df: pd.DataFrame):

    # Inicializar el StandardScaler
    scaler = StandardScaler()

    # Seleccionar las columnas que deseas estandarizar
    columns_to_scale = ['Salary', 'Salaries Reported']

    # Aplicar el estandarizador a las columnas seleccionadas
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])