#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:40:43 2024

@author: Estudiante
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

yacares=pd.read_csv("/home/Estudiante/Escritorio/Tp labo/yacares.csv")
datos_libreta=pd.read_csv("/home/Estudiante/Escritorio/Tp labo/datos_libreta_8023.csv")
def regresor_casero(X,Y):
    media_x =0
    media_y =0    
    a=0
    b=0
    for i in range(len(X)):
        media_x+=X.iloc[i]/len(X)
        media_y+=Y.iloc[i]/len(Y)
        
    for i in range(len(X)):
        a+=(X.iloc[i]-media_x)*(Y.iloc[i]-media_y)
        b+=(X.iloc[i]-media_x)*(X.iloc[i]-media_x)
    b1=a/b
    b0= media_y-b1*media_x
    
    return b0,b1
def recta(b0,b1,x):
    return b0+b1*x



b0,b1 =regresor_casero(yacares.iloc[:,0], yacares.iloc[:,1])



fig, ax =plt.subplots()
ax.scatter(yacares.iloc[:,0],yacares.iloc[:,1])
ax.plot( np.linspace(0, 1800 ), [recta(b0,b1,x) for x in np.linspace(0, 1800)])

b1 , b0 = regresor_casero(datos_libreta.iloc[:,0])