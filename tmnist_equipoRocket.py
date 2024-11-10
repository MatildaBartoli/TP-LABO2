"""
ALUMNOS: Emiliano Torres, Matilda Bartoli y Sofia Copaga
GRUPO : EQUIPO ROCKET

Este archivo.py contiene la resolución de los ejercicios propuestos en el TP2, separados en 5 secciones:
1) Importación de librerias
2) Funciones 
3) Análisis exploratorio de los datos
4) Clasificación binaria
5) Clasificación multiclase
    
Advertencia : el código tarda en correr alrededor de cinco minutos en su totalidad. 

"""

#%%===========================================================================
# IMPORTACIÓN DE LIBRERIAS
#=============================================================================
import pandas as pd
import numpy as np
from inline_sql import sql,sql_val
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split , KFold
from sklearn import tree
import seaborn as sns
import random as rn


#%%===========================================================================
# IMPORTACIÓN DE LA BASE DE DATOS
#=============================================================================

imagenes=pd.read_csv("/Users/Usuario/Downloads/TP-LABO2/TMNIST_Data.csv")


#%%===========================================================================
# FUNCIONES
#=============================================================================

#Toma un dataframe y devuelve una lista de las columnas que no suman 0
def columnas_relevantes(dataframe)->list:
    columnas_relevantes=[]
    for columna in dataframe.columns:
        if sum((dataframe[columna]).apply(int))!=0:
            columnas_relevantes.append(columna)    
    return columnas_relevantes

#Calculamos la exactitud como aciertos/total
def calcular_exactitud(prediccion,test)->float:
    aciertos=0
    for j in range(len(prediccion)):
        if(prediccion[j]==test.iloc[j]):
            aciertos+=1
    return aciertos/len(prediccion)

#Calcula la matriz de confusión
def matriz_de_confusion(prediccion,test):
    categorias_vistas=[]
    categorias_diferentes=0
    for i in range(len(prediccion)):
        if not(test.iloc[i] in categorias_vistas):
            categorias_vistas.append(test.iloc[i])
            categorias_diferentes+=1
    
    matriz=np.zeros([categorias_diferentes,categorias_diferentes])
    for i in range(len(prediccion)):
        matriz[test.iloc[i]][prediccion[i]]+=1 
        #la fila es del numero real y la columna la da la prediccion
    return matriz                
 
           
#%%===========================================================================
# ANÁLISIS EXPLORATORIO DE LOS DATOS
#=============================================================================

#Vemos las diferencias entre los números 2 y 8 promedios, dos números que consideramos similares
#Filtramos las filas del 2 y las del 8
imagenes_8=sql^"""SELECT * FROM imagenes WHERE labels=8"""
imagenes_2=sql^"""SELECT * FROM imagenes WHERE labels=2"""

#Retiramos las labels y la tipografía
imagenes_8=imagenes_8.iloc[:,2:]
imagenes_2=imagenes_2.iloc[:,2:]

#Inicializamos un par de arrays donde vamos a guardar los promedios
imagen_8_promedio=np.array([0 for _ in range(28*28)], dtype=float)
imagen_2_promedio=np.array([0 for _ in range(28*28)], dtype=float)

#Acumulamos todos los 2 y 8
for index, row in imagenes_8.iterrows():
    array=np.array(row)
    imagen_8_promedio+=array
    
for index, row in imagenes_2.iterrows():
    array=np.array(row)
    imagen_2_promedio+=array
    
#Dividimos por el total de imagenes de cada uno
imagen_8_promedio/=2990
imagen_2_promedio/=2990 

#Realizamos el gráfico para mostrar los resultados
fig, ax= plt.subplots(1,3)
ax[0].set_title("Imagen del 8 promedio")
ax[1].set_title("Imagen del 2 promedio")
ax[2].set_title("Diferencia entre ambas")
ax[0].imshow(imagen_8_promedio.reshape(28,28),cmap='gray')
ax[1].imshow(imagen_2_promedio.reshape(28,28),cmap='gray')
ax[2].imshow((abs(imagen_8_promedio-imagen_2_promedio)).reshape(28,28),cmap='gray')
ax[0].axis("off")
ax[1].axis("off")
ax[2].axis("off")
plt.tight_layout()
plt.show()

#%%Vemos unos ejemplos de las diferencias inter clases (nos centramos en la clase del número cero)

#Elegimos las filas de la clase 0
imagenes_0= sql^"""SELECT * FROM imagenes WHERE labels=0"""

#Separamos las imagenes de esas filas
X_0=imagenes_0.iloc[:,2:]

#Realizamos el grafico de 2 ceros para ilustrar las diferencias
fig, ax =plt.subplots(1,2)
ax[0].imshow(np.array(X_0.iloc[1,:]).reshape(28,28),cmap='gray')
ax[1].imshow(np.array(X_0.iloc[3,:]).reshape(28,28),cmap='gray')
ax[0].axis("off")
ax[1].axis("off")
plt.show()

#%%===========================================================================
# CLASIFICACIÓN BINARIA
#=============================================================================

#%% Filtramos las imagenes del 0 y el 1
imagenes_0= sql^"""SELECT * FROM imagenes WHERE labels=0"""
imagenes_1= sql^"""SELECT * FROM imagenes WHERE labels=1"""

#Unimos ambos dataframes
imagenes_binarias= sql^"""SELECT * FROM imagenes_0 UNION
                     SELECT * FROM imagenes_1"""
                     
#Separamos las imagenes de las labels
X_binario=imagenes_binarias.iloc[:,2:]
Y_binario=imagenes_binarias.iloc[:,1]

#Calculamos columnas no borde
lista_columnas_relevantes=columnas_relevantes(X_binario)

#%% Separamos los datos en test y train, entrenamos el modelo y calculamos la exactitud

#Tomamos las columnas distintas de 0 de X_binario 
X_binario_relevante=X_binario[lista_columnas_relevantes]

#Separamos en train y test (70% para train y 30% para test)
X_binario_train , X_binario_test, Y_binario_train , Y_binario_test = train_test_split(X_binario_relevante, Y_binario, test_size = 0.3,random_state=4) 


#%% Realizamos el KNN con atributos al azar entre 3 y 30

random_seed_sample=[7,3,19]

exactitud_promedio=[]

for r in range (3,32,2): #cantidad de atributos
    #lista donde vamos a guardar los resultados de exactitud
    lista_de_exactitud=[]
    for i in range(3): #hacemos 3 iteraciones para tomar promedio
    
        rn.seed(random_seed_sample[i]) #fijamos una semilla para la reproducibilidad
        #tomo r atributos relevantes
        atributos= rn.sample(lista_columnas_relevantes, r) 
        
        # selecciono esas columnas de los elementos de train
        X_train_sample=X_binario_train.iloc[:][atributos] 
        #seleccionamos los atributos
        X_binario_test_reducido=X_binario_test.iloc[:][atributos]
        #creamos el modelo
        model = KNeighborsClassifier(n_neighbors = 5) 
        
        model = model.fit(X_train_sample, Y_binario_train)   #entreno el modelo
        
        prediccion=model.predict(X_binario_test_reducido) #hago la predicción
        
        #calculo exactitud
        lista_de_exactitud.append(calcular_exactitud(prediccion,Y_binario_test)*100) 
        
    #Guardamos la exactitud promedio para cada cantidad de atributos
    exactitud_promedio.append(sum(lista_de_exactitud)/3)

#%% Graficamos promedio de exactitud

fig, ax= plt.subplots()
cantidad_de_atributos= [x for x in range(3,32,2)]
ax.plot(cantidad_de_atributos,exactitud_promedio, marker="o")
ax.set_xlabel("Cantidad de atributos")
ax.set_xticks([x for x in range(3,32,2)])
ax.set_ylabel("Exactitud(%)")
plt.ylim([60,100])
plt.grid()
plt.show()

#%%Calculo de exactitud para distinta cantidad de vecinos y atributos

#donde vamos a guardar la exactitud promedio para cada cantidad de vecinos y atributos
cantidad_vecinos_exactitud={}

for k in range(3,30,5): #k es la cantidad de vecinos
    #donde vamos a guardar la exactitud para cada cantidad de atributos
    exactitud_promedio=[]
    for r in range (3,32,2): #r es la cantidad de atributos
        
        lista_exactitud=[] #donde guardaremos la exactitud por modelo
        
        for i in range(3): #i representa el intento de entrenamiento
            #Fijamos una semilla para la reproducibilidad
            rn.seed(random_seed_sample[i])
            #tomo r atributos relevantes
            atributos= rn.sample(lista_columnas_relevantes, r) 
            # selecciono esas columnas de los elementos de train y test
            X_train_sample=X_binario_train.iloc[:][atributos] 
            
            X_binario_test_reducido=X_binario_test.iloc[:][atributos]
            
            model = KNeighborsClassifier(n_neighbors = k) #Creo el modelo con k vecinos
            
            model = model.fit(X_train_sample, Y_binario_train)   #entreno el modelo
            
            prediccion=model.predict(X_binario_test_reducido) #hago la prediccion
            #calculo la exactitud y la guardo
            lista_exactitud.append(calcular_exactitud(prediccion,Y_binario_test)*100)
        
        #tomo la exactitud promedio para valor de r y k
        exactitud_promedio.append(sum(lista_exactitud)/3) 
        
    #luego la guardo en el diccionario
    cantidad_vecinos_exactitud[k]=exactitud_promedio
    
    
#%%Gráficos comparando cantidad de atributos, cantidad de vencinos y exactitud
fig, ax= plt.subplots()
cantidad_de_atributos= [x for x in range(3,32,2)]
claves=list(cantidad_vecinos_exactitud.keys())
color=["red","blue","green","m","crimson","darkseagreen","yellow"]
for i in range(6):
    ax.plot(cantidad_de_atributos,cantidad_vecinos_exactitud[claves[i]],color=color[i],marker="o" ,label=f"k={claves[i]}")
    ax.legend()
    ax.set_xlabel("Cantidad de atributos")
    ax.set_xticks([x for x in range(3,32,2)])
    ax.set_ylabel("Exactitud (%)")
    plt.ylim([60,100])
    
plt.grid() 
plt.show()   


#%%===========================================================================
# CLASIFICACIÓN MULTICLASE 
#=============================================================================

#%% Separamos los datos
X=imagenes.iloc[:,2:]

#Tomamos columnas relevantes
X_relevante=X[columnas_relevantes(X)]
Y=imagenes.iloc[:,1]

#Separamos en desarrollo (dev) y held out
X_dev, X_held_out, Y_dev, Y_held_out= train_test_split(X_relevante,Y,test_size=0.1, random_state=21)


#%% Entrenamos el modelo
#Separamos dev en train y test
X_train , X_test, Y_train, Y_test = train_test_split(X_dev,Y_dev, test_size=0.3, random_state=42)

#guardamos la exactitud para cada profundidad
lista_exactitud=[]
for k in range(1,11): #k es la profundidad maxima
    #iniciamos el modelo con criterio entropia
    model=tree.DecisionTreeClassifier(criterion="entropy",max_depth=k)
    model=model.fit(X_train, Y_train) #entrenamos el modelo
    prediccion=model.predict(X_test) #hacemos la prediccion
    #calculamos y guardamos la exactitud
    lista_exactitud.append(calcular_exactitud(prediccion,Y_test)*100)

#%% Gráfico de exactitud para arboles con distinta profundidad
fig, ax= plt.subplots()
profundidad= [x for x in range(1,11)]
ax.plot(profundidad,lista_exactitud,marker="o")
ax.set_xlabel("Profundidad máxima")
ax.set_xticks([x for x in range(1,11)])
ax.set_ylabel("Exactitud (%)")
ax.set_yticks([y for y in range(0,110,10)])
plt.ylim([0,100])
plt.grid()
plt.show()
   
#%%Hacemos la variación de criterio y profundidad con el K-folding

criterio=["entropy","gini"] 
kf = KFold(n_splits=5) #realizamos el k-fold
resultados_por_fold={} #donde guardaremos los resultados por fold
#tomamos la decision de reconstruir el mejor modelo porque aumenta mucho el tiempo
#de espera hacer la matriz de confusion con cada uno de los mejores modelos
folds={} #donde guardaremos los folds para poder reconstruir el mejor modelo
fold=1

#Hacemos el k-folding
for train_index, test_index in kf.split(X_dev):
    #separamos los datos
    X_train=X_dev.iloc[train_index,:] 
    Y_train=Y_dev.iloc[train_index]
    X_test=X_dev.iloc[test_index,:]
    Y_test=Y_dev.iloc[test_index]

    exactitud_por_criterio={} #vamos a guardar los resultados por criterio
    for c in range(2):
        lista_exactitud=[] #vamos a guardar la exactitud para las 10 profunidades
        for k in range(1,11):
            #variamos criterio y la profundidad
            model=tree.DecisionTreeClassifier(criterion=criterio[c],max_depth=k) 
            
            model=model.fit(X_train, Y_train) #entrenamos el modelo
            prediccion=model.predict(X_test) #hacemos las predicciones
            lista_exactitud.append(calcular_exactitud(prediccion,Y_test)*100) 
            #calulamos y agregamos la exactidud a la lista
            
        #guardamos las 10 exactitudes para cada criterio
        exactitud_por_criterio[criterio[c]]=lista_exactitud
    resultados_por_fold[fold]=exactitud_por_criterio #guardamos los resultados por fold
    #guardamos los folds para reconstruir a nuestro mejor modelo luego
    folds[fold]=(train_index, test_index)
    fold+=1


#%%Elejimos el mejor modelo
maximo=0 
for fold in range(1,6): #revisamos los 5 folds
    diccionario_exactitud_del_fold=resultados_por_fold[fold]
    for criterio in diccionario_exactitud_del_fold.keys(): #revisamos por los 2 criterios
        lista_exactitud=diccionario_exactitud_del_fold[criterio]
        for i in range(10): # revisamos las 10 profundidades
            if lista_exactitud[i]>maximo:
                #guardamos los parametros que consiguen las maximas exactitudes
                maximo=lista_exactitud[i]
                mejor_profundidad=i+1
                mejor_fold=fold
                mejor_criterio=criterio

#Vemos su performance
model=tree.DecisionTreeClassifier(criterion=mejor_criterio,max_depth=mejor_profundidad)
model=model.fit(X_dev.iloc[folds[mejor_fold][0],:],Y_dev.iloc[folds[mejor_fold][0]])
prediccion=model.predict(X_dev.iloc[folds[mejor_fold][1],:])

#Calculamos matriz de confusión y exactitud
matriz_de_confusion_mejor_modelo=matriz_de_confusion(prediccion,Y_dev.iloc[folds[mejor_fold][1]])
exactitud_mejor_modelo=calcular_exactitud(prediccion,Y_dev.iloc[folds[mejor_fold][1]])*100

#%% Exactitud promedio del k-folding
exactitud_promedio_k_folding=0
for fold in range(1,6): #revisamos los 5 folds
    diccionario_exactitud_del_fold=resultados_por_fold[fold]
    for criterio in diccionario_exactitud_del_fold.keys(): #revisamos por los 2 criterios
        lista_exactitud=diccionario_exactitud_del_fold[criterio]
        for i in range(10): # revisamos las 10 profundidades
            exactitud_promedio_k_folding+=lista_exactitud[i]/100 #hacemos el promedio sobre los 100 modelos que entrenamos

#%%Vemos la exactitud de los mejores modelos de cada criterio por folder
mejores_modelos_por_fold={}
for fold in range(1,6): #revisamos los 5 folds
    diccionario_exactitud_del_fold=resultados_por_fold[fold]
    mejores_modelos_por_criterio={}
    for criterio in diccionario_exactitud_del_fold.keys(): #revisamos por los 2 criterios
        lista_exactitud=diccionario_exactitud_del_fold[criterio]
        mejores_modelos_por_criterio[criterio]=max(lista_exactitud)
    mejores_modelos_por_fold[fold]=mejores_modelos_por_criterio
    
#Conseguimos la exactitud para cada criterio
exactitud_gini=[mejores_modelos_por_fold[fold]['gini'] for fold in mejores_modelos_por_fold.keys()]
exactitud_entropia=[mejores_modelos_por_fold[fold]['entropy'] for fold in mejores_modelos_por_fold.keys()]
#hacemos la resta
diferencia=[exactitud_gini[i]-exactitud_entropia[i] for i in range(len(exactitud_entropia))]

#Separamos por los que fue mayor gini o entropia
#notemos que es una lista vacia,significa que entropia fue mejor en cada fold
mayor_gini=[x for x in diferencia if x>0]
mayor_entropia=[abs(x) for x in diferencia if x<0]

#Graficamos
fig, ax=plt.subplots()
ancho_barra=0.3
ax.bar([1,2,3,4,5],mayor_entropia)
ax.set_ylabel("Diferencias de exactitud (%)")
ax.set_xlabel("Numero de folder")
ax.set_xticks([1,2,3,4,5])
plt.show()
#%%Entrenamos al modelo con el held-out

model=tree.DecisionTreeClassifier(criterion=mejor_criterio,max_depth=mejor_profundidad)
model=model.fit(X_dev,Y_dev)
prediccion=model.predict(X_held_out) #hacemos la prediccion

#Calculamos la exactitud y matriz de confusion con el conjunto held out
exactitud_held_out=calcular_exactitud(prediccion,Y_held_out)*100
matriz_de_confusion_held_out=matriz_de_confusion(prediccion, Y_held_out)
