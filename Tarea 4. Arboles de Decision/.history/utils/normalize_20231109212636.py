

def muestreo_aleatorio(df, column_name):
    # Identificamos los valores no cero en la columna 'column_name'
    non_zero_BloodPressure = df['column_name'][df['column_name'] != 0]

    # Contamos el número de ceros en la columna 'BloodPressure'
    num_zeros_BloodPressure = (df['column_name'] == 0).sum()

    # Tomamos una muestra aleatoria de los valores no cero de 'BloodPressure'
    random_sample_BloodPressure = non_zero_BloodPressure.sample(n=num_zeros_BloodPressure, replace=True, random_state=1)

    # Reemplazamos los valores cero en 'BloodPressure' con la muestra aleatoria
    df.loc[df['column_name'] == 0, 'column_name'] = random_sample_BloodPressure.values

    # Verificamos si el reemplazo fue exitoso contando nuevamente el número de ceros
    num_zeros_after_replacement = (df['column_name'] == 0).sum()
