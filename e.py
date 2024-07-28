import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp
from autograd import jacobian


# Constantes
pKa_acetico = 4.756
pKa_amoniaco = 9.25

Ka_acetico = 10**(pKa_acetico)
Kb_amoniaco = 10**(-pKa_amoniaco)

Kw = 10**(-14) # Constante de disociación del agua

N = 100 # Número de puntos a lo largo de la columna


# Definir las concentraciones lineales
conc_acido = np.linspace(1e-3, 10e-3, N)  # 1 mM a 10 mM
conc_amoniaco = np.linspace(10e-3, 1e-3, N)  # 10 mM a 1 mM


def electroneutralidad(H, i):
    NH4 = (conc_amoniaco[i] * H) / (Kb_amoniaco + H)
    CH3COO = conc_acido[i] / ((Ka_acetico * H) + 1)
    OH = Kw / H
    return H - OH + NH4 - CH3COO


def renglonMatriz(i):
    # Crear un arreglo de 100 ceros
    arreglo = np.zeros(100, dtype=object)
    if 0 <= i-1 < 100:
        f1 = lambda h1=vars[i-1]: 1/(1+abs(i-1))*electroneutralidad(h1, i-1)
        arreglo[i-1] = f1
    if 0 <= i < 100:
        f2 = lambda h2=vars[i]: 1/(1+abs(i))*electroneutralidad(h2, i)
        arreglo[i] = f2
    if 0 <= i+1 < 100:
        f3 = lambda h3=vars[i+1]: 1/(1+abs(i+1))*electroneutralidad(h3, i+1)
        arreglo[i+1] = f3

    return arreglo
    

def matrizA(N):
    matriz = np.zeros((N,N), dtype=object)
    for i in range(N):
        matriz[i] = renglonMatriz(i)
    return matriz

resultado_matriz = matrizA(N)

#for fila in resultado_matriz:
     #print(fila)

def electroneutralidad_primera_derivada(H, i):
    NH4 = (conc_amoniaco[i] * Kb_amoniaco) / (H + Kb_amoniaco)**2
    CH3COO = -(conc_acido[i] * Ka_acetico) / ((Ka_acetico * H) + 1)**2
    OH = - Kw / H**2
    return 1 - OH + NH4 - CH3COO

def electroneutralidad_segunda_derivada(H, i):
    NH4 = -(2 * conc_amoniaco[i] * Kb_amoniaco) / (H + Kb_amoniaco)**3
    CH3COO = (2 * conc_acido[i] * Ka_acetico**2) / ((Ka_acetico * H) + 1)**3
    OH = (2 * Kw) / H**3
    return - OH + NH4 - CH3COO


# Inicialización de variables
initial_1 = 1e-10
initial_2 = 1e-9

ph_values = []