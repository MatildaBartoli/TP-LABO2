#%%importamos librerias
import pandas as pd
import numpy as np
from inline_sql import sql,sql_val
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

def matriz_de_confusion(Y_test,prediccion):
    categorias_vistas=[]
    categorias_diferentes=0
    for i in range(len(prediccion)):
        if not(Y_test.iloc[i] in categorias_vistas):
            categorias_vistas.append(Y_test.iloc[i])
            categorias_diferentes+=1
    
    matriz=np.zeros([categorias_diferentes,categorias_diferentes])
    for i in range(len(prediccion)):
        matriz[Y_test.iloc[i]][prediccion[i]]+=1 
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
tercer_opcion=((muestra_0).apply(float)-(muestra_1).apply(float)).apply(abs)
ax[2].imshow(np.array(tercer_opcion).reshape(28,28),cmap='gray')
ax[0].imshow(np.array(muestra_1.apply(float)).reshape(28,28),cmap='gray')
ax[1].imshow(np.array(muestra_0.apply(float)).reshape(28,28),cmap='gray')

#%% filtramos las imagenes del 0 y el 1

imagenes_0= sql^"""SELECT * FROM imagenes WHERE labels=0"""
imagenes_1= sql^"""SELECT * FROM imagenes WHERE labels=1"""

imagenes_1_0= sql^"""SELECT * FROM imagenes_0 UNION
                     SELECT * FROM imagenes_1"""
                     
X_10=imagenes_1_0.iloc[:,2:]
Y_10=imagenes_1_0.iloc[:,1]

#calculamos columnas no borde
lista_columnas_relevantes=columnas_relevantes(X_10)
    
#%% Separamos los datos en test y train, entrenamos el modelo y calculamos la exactitud
X_10_train , X_10_test, Y_10_train , Y_10_test = train_test_split(X_10, Y_10, test_size = 0.3,random_state=4) # 70% para train y 30% para test
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
        
        X_train_sample=X_10_train.iloc[:][atributos] # selecciono esas columnas de los elementos de train
        
        X_10_test_reducido=X_10_test.iloc[:][atributos]
        
        model = KNeighborsClassifier(n_neighbors = 5) # Creo el modelo en abstracto
        
        model = model.fit(X_train_sample, Y_10_train)   #entreno el modelo
        
        prediccion=model.predict(X_10_test_reducido) #hago la prediccion
        #calculo exactitud
        aciertos=0
        for j in range(len(prediccion)):
            if(prediccion[j]==Y_10_test.iloc[j]):
                aciertos+=1
        #sabemos que se calcula con la matriz de confusion
        #pero por un tema de complejidad temporal lo calculamos asi
        lista_de_exactitud.append(aciertos/len(prediccion)*100) 
                                                                                            
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
            
            X_train_sample=X_10_train.iloc[:][atributos] # selecciono esas columnas de los elementos de train
            
            X_10_test_reducido=X_10_test.iloc[:][atributos]
            
            model = KNeighborsClassifier(n_neighbors = k) # Creo el modelo en abstracto
            
            model = model.fit(X_train_sample, Y_10_train)   #entreno el modelo
            
            prediccion=model.predict(X_10_test_reducido) #hago la prediccion
            
            lista_de_exactitud.append(calcular_exactitud(prediccion,Y_10_test)*100)
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

X=imagenes.iloc[:,2:]
Y=imagenes.iloc[:,1]

X_train , X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.35 ,random_state=42)
X_test, X_held_out, Y_test, Y_held_out= train_test_split(X_test,Y_test,test_size=0.005, random_state=21)
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
   
#%% Cambiar hiperparametros
criterio=["entropy","gini"]
for i in range(len(criterio)):
    for j in range (1,11):
#que imperparametros podemos variar? pensamos en profundidad y criterio









