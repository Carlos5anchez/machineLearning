from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize(df: pd.DataFrame):

    # Inicializar el StandardScaler
    scaler = StandardScaler()
    dftranformed = scaler.fit_transform(df)
    return dftranformed