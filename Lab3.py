# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:01:10 2021

@author: Diego Morales
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from scipy import fftpack

#Creamos una senal

#Definir la frecuencia
f = 10
#Cantidad de mediciones por segundo
f_s = 100

#Creamos la senal de prueba
t = np.linspace(0, 2, 2 * f_s, endpoint=False)
x = np.sin(f * 2 * np.pi * t) 

#Graficamos la senal
plt.plot(t, x)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud de la senal');
plt.grid()
plt.show()
print('Esta es la señal original')

#Vamos a implementar la transformada de Fourier tradicional para distintas 
#longitudes y vamos a medir el tiempo para cada longitud

N = np.linspace(1,len(t),int(len(t)/5)) #definicion de cada longitud
t0 = []
X =[]
T = []

for i in range(len(N)):
    t0.append(time.time()) #almacenamos la hora inicial para cada longitud
    F = 0
#vamos a calcular la transformada de Fourier tradicional
    for k in range(int(N[i])):
        for j in range(int(N[i])):
            F += x[j]*np.exp(-(2*np.pi*j*k)/N[i])
        X.append(F)
    T.append(time.time()-t0[i])        
    
#Graficamos el comportamiento
    
plt.plot(N,T)
plt.xlabel('Puntos Considerados')
plt.ylabel('Tiempo')
plt.grid()
plt.show()

logT = np.log(T)
logN = np.log(N).reshape((-1,1))

model = LinearRegression().fit(logN,logT)
R2 = model.score(logN,logT)
predict = model.intercept_*np.exp(model.coef_*N)

plt.loglog(N,T)
plt.loglog(N,predict)
plt.xlabel('Puntos Considerados')
plt.ylabel('Tiempo')
plt.grid()
plt.show()

print('intercepto: ', model.intercept_)
print('coeficiente: ', model.coef_)
print('R^2: ',R2)

Npredict = 20000

print('Tiempo predicho para ', Npredict, 'puntos es ', 
      np.exp(model.coef_*np.log(Npredict)+model.intercept_)/60, 
      ' minutos.')
print('Esta es la transformada normal de Fourier')

#Ahora con la FFT
t0F = []
TF = []
#Ahora calculamos la transformada rapida de Fourier
for i in range(len(N)):
    t0F.append(time.time()) #almacenamos la hora inicial para cada longitud
    XF = fftpack.fft(x[:int(N[i])])
    TF.append(time.time()-t0F[i])

#Graficamos el comportamiento
    
plt.plot(N,TF)
plt.xlabel('Puntos Considerados')
plt.ylabel('Tiempo')
plt.ylim(-0.1,0.1)
plt.grid()
plt.show()
print('Esta es la trasnformada rápida de Fourier')