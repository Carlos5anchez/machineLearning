from types import SimpleNamespace

def getConfig():
    return SimpleNamespace(
        model="informer",  # Opciones: [informer, informerstack, informerlight(TBD)]
        data="custom",  # data 
        root_path="./data/ETT/",  # Folder del archivo de datos
        data_path="DMN_Report_29-All.csv",  # Nombre del archivo de datos
        features="MS",  # options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        target="Lactose", # Característica objetivo en la tarea  de predicción
        freq="w", #frecuencia para codificación de funciones de tiempo, opciones:[s:segundo, t:minuto, h:hora, d:diario, b:días hábiles, w:semanal, m:mensual], también puede usar una frecuencia más detallada como 15 minutos o 3 horas '
        detail_freq="w", #Tiene que ser igual a la anterior
        checkpoints="./checkpoints/",
        seq_len=96, # Longitud de la secuencia de entrada del codificador Informer, cantidad para 
        label_len=48, # Longitud del token de inicio del decodificador Informer
        pred_len=30, # Longitud de la secuencia de predicción (Cuanto quiero a futuro para predecir )
        enc_in=15, # Tamaño de entrada del codificador
        dec_in=15, # Tamaño de entrada del decodificador
        c_out=1,  # Tamaño de salida
        d_model=512, # Dimensión del modelo
        n_heads=8, # Número de cabezas
        e_layers=2, # Número de capas del codificador
        d_layers=1, # Número de capas del decodificador
        s_layers="3,2,1", # Número de capas de la pila del codificador
        d_ff=2048, # Dimensión de la red neuronal (fcn)
        factor=5, # Factor de atención probsparse
        padding=0, # Tipo de relleno
        distil=True, # Si se debe usar destilación en el codificador, usar este argumento significa no usar destilación
        dropout=0.05,
        attn="prob", # Atención utilizada en el codificador, opciones:[prob, full]
        embed="timeF",  # Codificación de características de tiempo, opciones:[timeF, fixed, learned]
        activation="gelu", # Función de activación
        output_attention=False, # Si se debe generar atención en el codificador
        do_predict=False, # Si se debe predecir datos futuros invisibles
        mix=True, # Usar atención mixta en el decodificador generativo
        cols='',  # Asegúrate de proporcionar una lista adecuada si es necesario
        num_workers=0,
        itr=6,#=========Tiempos de experimentos========
        train_epochs=64, #======================= Epocas de entrenamiento
        batch_size=64, # Tamaño del lote de datos de entrada de entrenamiento
        patience=3,
        learning_rate=0.0001,
        des="Pruebas", #Experiment description 
        loss="mse",
        lradj="type1",
        use_amp=False,
        inverse=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0,1,2,3"
    )