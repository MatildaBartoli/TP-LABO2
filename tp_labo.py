#%%importamos librerias
import pandas as pd
import numpy as np
from inline_sql import sql,sql_val
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split , KFold
from sklearn import tree
import seaborn as sns
import random as rn

#%%Importamos base de datos
#dejo comentada mi carpeta xD asi no busco la ruta en cada pull
#Emi
#".\\TMNIST_Data.csv"
#Sofi
#"C:/Users/copag/Desktop/TP2_labo/TMNIST_Data.csv"
imagenes=pd.read_csv(".\\TMNIST_Data.csv")
#%%funciones  
def columnas_relevantes(dataframe)->list:
    columnas_relevantes=[]
    for columna in dataframe.columns:
        if sum((dataframe[columna]).apply(int))!=0:
            columnas_relevantes.append(columna)    
    return columnas_relevantes

def calcular_exactitud(prediccion,test)->float:
    aciertos=0
    for j in range(len(prediccion)):
        if(prediccion[j]==test.iloc[j]):
            aciertos+=1
    return aciertos/len(prediccion)

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
        #aprovechamos que son numeros, sino habria que mapearlos con un diccionario
    return matriz                
            
#%% Analisis exploratorio atributos relevantes

#tomemos una muestra por ejemplo los 2 primeros
muestra=sql^"""SELECT * FROM imagenes LIMIT 2"""
etiquetas_muestra=muestra.iloc[:,1]
datos_muestra=muestra.iloc[:,2:]

#Notemos que en las imagenes las primeras y ultimas columnas de pixeles son un borde negro
#no podemos sacar diferencias de ahi
#Entonces las diferencias estan en los pixeles centrales, los distintos de 0
#veamos las columnas distintas de 0
lista_columnas_relevantes=columnas_relevantes(datos_muestra)


#vemos que redujimos el problema a mas de la mitad, pasamos de 784 columnas a 236
#nos interesan ver las diferencias entre ambas fotos
#eso lo podemos ver en la resta, ya que los pixeles del mismo color se anulan al restarlos, 
#entonces solo vemos las diferencias entre las imagenes
muestra_0=muestra.iloc[0,2:]
muestra_1=muestra.iloc[1,2:]


#%% Vemos las diferencias entre 2 y 8
fig, ax= plt.subplots(1,3)
ax[0].set_title("Imagen de un 8")
ax[1].set_title("Imagen de un 2")
ax[2].set_title("diferencia entre ambas")
imagenes_8=sql^"""SELECT * FROM imagenes WHERE labels=8"""
imagenes_2=sql^"""SELECT * FROM imagenes WHERE labels=2"""
imagenes_8=imagenes_8.iloc[:,2:]
imagenes_2=imagenes_2.iloc[:,2:]
imagen_8_promedio=np.array([0 for _ in range(28*28)], dtype=float)
imagen_2_promedio=np.array([0 for _ in range(28*28)], dtype=float)
for index, row in imagenes_8.iterrows():
    array=np.array(row)
    imagen_8_promedio+=array
    
for index, row in imagenes_2.iterrows():
    array=np.array(row)
    imagen_2_promedio+=array

imagen_8_promedio/=2990
imagen_2_promedio/=2990 
ax[0].imshow(imagen_8_promedio.reshape(28,28),cmap='gray')
ax[1].imshow(imagen_2_promedio.reshape(28,28),cmap='gray')
ax[2].imshow((abs(imagen_8_promedio-imagen_2_promedio)).reshape(28,28),cmap='gray')
plt.tight_layout()
#%%veamos unos ejemplos de las diferencias inter clases
imagenes_0= sql^"""SELECT * FROM imagenes WHERE labels=0"""
X_0=imagenes_0.iloc[:,2:]
fig, ax =plt.subplots(1,2)
ax[0].imshow(np.array(X_0.iloc[1,:]).reshape(28,28),cmap='gray')
ax[1].imshow(np.array(X_0.iloc[3,:]).reshape(28,28),cmap='gray')


#%% filtramos las imagenes del 0 y el 1

imagenes_0= sql^"""SELECT * FROM imagenes WHERE labels=0"""
imagenes_1= sql^"""SELECT * FROM imagenes WHERE labels=1"""

imagenes_1_0= sql^"""SELECT * FROM imagenes_0 UNION
                     SELECT * FROM imagenes_1"""
                     
X_binario=imagenes_1_0.iloc[:,2:]
Y_binario=imagenes_1_0.iloc[:,1]

#calculamos columnas no borde
lista_columnas_relevantes=columnas_relevantes(X_binario)

#%% Separamos los datos en test y train, entrenamos el modelo y calculamos la exactitud
X_binario_train , X_binario_test, Y_binario_train , Y_binario_test = train_test_split(X_binario, Y_binario, test_size = 0.3,random_state=4) 
# 70% para train y 30% para test
#KNN = K nearest neighbors
#Tomemos atributos random peroo dentro de las columnas relevantes
#pruebas con 3 atributos

random_seed_sample=[7,3,19]

exactitud_promedio=[]

for k in range (3,32,2):

    lista_de_exactitud=[]
    for i in range(3):
        rn.seed(random_seed_sample[i])
        atributos= rn.sample(lista_columnas_relevantes, k) #tomo k atributos relevantes
        
        X_train_sample=X_binario_train.iloc[:][atributos] # selecciono esas columnas de los elementos de train
        
        X_binario_test_reducido=X_binario_test.iloc[:][atributos]
        
        model = KNeighborsClassifier(n_neighbors = 5) # Creo el modelo en abstracto
        
        model = model.fit(X_train_sample, Y_binario_train)   #entreno el modelo
        
        prediccion=model.predict(X_binario_test_reducido) #hago la prediccion
        #calculo exactitud
        lista_de_exactitud.append(calcular_exactitud(prediccion,Y_binario_test)*100) 
                                                                                            
    exactitud_promedio.append(sum(lista_de_exactitud)/3)

#%% grafiquemos promedio de exactitud

fig, ax= plt.subplots()
cantidad_de_atributos= [x for x in range(3,32,2)]
ax.plot(cantidad_de_atributos,exactitud_promedio, marker="o")
ax.set_xlabel("cantidad de atributos")
ax.set_xticks([x for x in range(3,32,2)])
ax.set_ylabel("porcentaje de exactitud")
plt.ylim([70,100])
plt.grid()
#Este grafico tiene k_neighbours=5
#%%Calculo de exactitud para distinta cantidad de vecinos y atributos

cantidad_vecinos_exactitud={}
for k in range(3,100,20):
    exactitud_promedio=[]
    for r in range (3,32,2):
        lista_de_exactitud=[]
        for i in range(3):
            rn.seed(random_seed_sample[i])
            atributos= rn.sample(lista_columnas_relevantes, r) #tomo r atributos relevantes
            
            X_train_sample=X_binario_train.iloc[:][atributos] # selecciono esas columnas de los elementos de train
            
            X_binario_test_reducido=X_binario_test.iloc[:][atributos]
            
            model = KNeighborsClassifier(n_neighbors = k) # Creo el modelo en abstracto
            
            model = model.fit(X_train_sample, Y_binario_train)   #entreno el modelo
            
            prediccion=model.predict(X_binario_test_reducido) #hago la prediccion
            
            lista_de_exactitud.append(calcular_exactitud(prediccion,Y_binario_test)*100)
        exactitud_promedio.append(sum(lista_de_exactitud)/3)
    
    cantidad_vecinos_exactitud[k]=exactitud_promedio
#%%Graficos comparando cantidad de atributos, cantidad de vencinos y exactitud
fig, ax= plt.subplots()
cantidad_de_atributos= [x for x in range(3,32,2)]
claves=list(cantidad_vecinos_exactitud.keys())
color=["red","blue","green","m","crimson","darkseagreen"]
for i in range(5):
    ax.plot(cantidad_de_atributos,cantidad_vecinos_exactitud[claves[i]],color=color[i],marker="o" ,label=f"k={claves[i]}")
    ax.legend()
    ax.set_xlabel("cantidad de atributos")
    ax.set_xticks([x for x in range(3,32,2)])
    ax.set_ylabel("porcentaje de exactitud (%)")
    plt.ylim([70,100])
    plt.grid()
    
#%% Clasificación multiclase con arboles
#%% Separamos los datos

X=imagenes.iloc[:,2:]
Y=imagenes.iloc[:,1]

X_dev, X_held_out, Y_dev, Y_held_out= train_test_split(X,Y,test_size=0.1, random_state=21)
#%% Entrenamos el modelo
X_train , X_test, Y_train, Y_test = train_test_split(X_dev,Y_dev, test_size=0.3, random_state=42)
lista_exactitud=[]
for k in range(1,11):
    model=tree.DecisionTreeClassifier(criterion="entropy",max_depth=k)
    model=model.fit(X_train, Y_train)
    prediccion=model.predict(X_test)
    lista_exactitud.append(calcular_exactitud(prediccion,Y_test)*100)
#deberiamos considerar la precision promedio aun?
a=matriz_de_confusion(Y_test,prediccion)

#%% Grafico de exacctitud para arboles con distinta profundidad
fig, ax= plt.subplots()
profundidad= [x for x in range(1,11)]
ax.plot(profundidad,lista_exactitud,marker="o")
ax.set_xlabel("Profundidad maxima")
ax.set_xticks([x for x in range(1,11)])
ax.set_ylabel("porcentaje de exactitud (%)")
ax.set_yticks([y for y in range(0,110,10)])
plt.ylim([0,100])
plt.grid()
   
#%%time Hacemos la variacion de parametros con el K-folding

criterio=["entropy","gini"]
kf = KFold(n_splits=5)
resultados_por_fold={} 
folds={}
fold=1
#hacemos el k-fold
for train_index, test_index in kf.split(X_dev):
    X_train=X_dev.iloc[train_index,:] 
    Y_train=Y_dev.iloc[train_index]
    X_test=X_dev.iloc[test_index,:]
    Y_test=Y_dev.iloc[test_index]

    exactitud_por_criterio={} #vamos a guardar los resultados por criterio
    for c in range(2):
        lista_exactitud=[] #vamos a guardar la exactitud para las 10 profunidades
        for k in range(1,11):
            model=tree.DecisionTreeClassifier(criterion=criterio[c],max_depth=k) #variamos criterio
            model=model.fit(X_train, Y_train) #entrenamos el modelo
            prediccion=model.predict(X_test) #hacemos las predicciones
            lista_exactitud.append(calcular_exactitud(prediccion,Y_test)*100) 
            #calulamos y agregamos la exactidud a la lista
        
        exactitud_por_criterio[criterio[c]]=lista_exactitud
    resultados_por_fold[fold]=exactitud_por_criterio
    folds[fold]=(train_index, test_index)#guardamos los folds para reconstruir a nuestro mejor modelo luego
    fold+=1


#%%elejimos el mejor modelo
maximo=0
for fold in range(1,6): #revisamos los 5 folds
    diccionario_exactitud_del_fold=resultados_por_fold[fold]
    for criterio in diccionario_exactitud_del_fold.keys(): #revisamos por los 2 criterios
        lista_exactitud=diccionario_exactitud_del_fold[criterio]
        for i in range(10): # revisamos las 10 profundidades
            if lista_exactitud[i]>maximo:
                maximo=lista_exactitud[i]
                mejor_profundidad=i+1
                mejor_fold=fold
                mejor_criterio=criterio
                
#veamos su performance
model=tree.DecisionTreeClassifier(criterion=mejor_criterio,max_depth=mejor_profundidad)
model=model.fit(X_dev.iloc[folds[mejor_fold][0],:],Y_dev.iloc[folds[mejor_fold][0]])
prediccion=model.predict(X_dev.iloc[folds[mejor_fold][1],:])
matriz_de_confusion_mejor_modelo=matriz_de_confusion(prediccion,Y_dev.iloc[folds[mejor_fold][1]])
exactitud_mejor_modelo=calcular_exactitud(prediccion,Y_dev.iloc[folds[mejor_fold][1]])*100
#%%veamos la exactidud de los mejores modelos de cada criterio por folder
mejores_modelos_por_fold={}
for fold in range(1,6): #revisamos los 5 folds
    diccionario_exactitud_del_fold=resultados_por_fold[fold]
    mejores_modelos_por_criterio={}
    for criterio in diccionario_exactitud_del_fold.keys(): #revisamos por los 2 criterios
        lista_exactitud=diccionario_exactitud_del_fold[criterio]
        mejores_modelos_por_criterio[criterio]=max(lista_exactitud)
    mejores_modelos_por_fold[fold]=mejores_modelos_por_criterio

exactitud_gini=[mejores_modelos_por_fold[fold]['gini'] for fold in mejores_modelos_por_fold.keys()]
exactitud_entropia=[mejores_modelos_por_fold[fold]['entropy'] for fold in mejores_modelos_por_fold.keys()]

diferencia=[exactitud_gini[i]-exactitud_entropia[i] for i in range(len(exactitud_entropia))]
mayor_gini=[x for x in diferencia if x>0]
mayor_entropia=[abs(x) for x in diferencia if x<0]
fig, ax=plt.subplots()
ancho_barra=0.3
ax.bar([1,2,3,4,5],mayor_entropia)
ax.set_ylabel("Diferencias de exactitud")
ax.set_xlabel("Numero de folder")
ax.set_xticks([1,2,3,4,5])
#%%Entrenamos al modelo con el held-out
model=tree.DecisionTreeClassifier(criterion=mejor_criterio,max_depth=mejor_profundidad)
model=model.fit(X_dev,Y_dev)
prediccion=model.predict(X_held_out)
matriz_de_confusion_held_out= matriz_de_confusion(prediccion,Y_held_out)
exactitud_held_out= calcular_exactitud(prediccion,Y_held_out)*100



