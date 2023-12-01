import torch

from exp.exp_informer import Exp_Informer
from config import getConfig

args=getConfig()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = { # dataset_name: [data_path, target, (num_encoder, num_decoder, num_features)]
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'DMN_Report_29_Processed':{'data':'DMN_Report_29_Processed.csv','T':'Lactose','M':[14,14,14],'S':[1,1,1],'MS':[14,14,1]},
}
if args.data in data_parser.keys(): # Si existe el dataset en el diccionario
    data_info = data_parser[args.data] # Se obtiene la información del dataset por la key (nombre del dataset)
    args.data_path = data_info['data'] # Se obtiene la ruta del dataset
    args.target = data_info['T'] # Se obtiene el target del dataset 
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]  #Multivariable , Univariable, Multivariable-Univariable (M,S,MS)


#Por cada capa de la red se obtiene el número de capas (se quitan los espacios y se separan por comas) "3,2,1" -> [3,2,1]
args.s_layers = [int(capa) for capa in args.s_layers.replace(' ','').split(',')] # Se obtiene el número de capas de la red
args.detail_freq = args.freq  #Se obtiene la frecuencia de los datos [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly] 
args.freq = args.freq[-1:]  #Se obtiene la frecuencia de los datos (s,t,h,d,b,w,m) ultimo caracter  (Defaul: h)

print('⚙️ =============( Lista de Argumentos )========= ⚙️:')
print(args)
print('⚙️ ============================= ⚙️')


Exp = Exp_Informer # Se crea el objeto de la clase Exp_Informer

# Guardado de los resultados
for ii in range(args.itr): # Se itera sobre el número de experimentos Defaul: 2
    #Por cada experimento se crea una carpeta
    #Configurando los resultados de los experimentos
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)

    # Se crea el objeto de la clase Exp_Informer con los argumentos
    exp = Exp(args) 
    
    
    print('================🏋️‍♂️  ( Entrenando Modelo )  🏋️‍♂️ ==============')
    print('Archivo: {}'.format(setting))
    exp.train(setting)
    
    print('================🔥 ( Probando Modelo )  🔥 ==============')
    print('Archivo: {}'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('================ 🔔( Prediccion con Modelo ) 🔔==============')
        print('Archivo: {}'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache() # Se libera la memoria de la GPU
    print("👍=============================")