import autograd.numpy as np
from autograd import jacobian
from scipy.optimize import fsolve

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

# Generar el sistema de ecuaciones
def sistema_ecuaciones(vars):
    n = len(vars)
    ecuaciones = []

    for i in range(n):
        if i == 0:
            ecuacion = (electroneutralidad(vars[i], i) * 1/1+abs(i)) + (electroneutralidad(vars[i+1], i+1) * 1/1+abs(i+1))
        elif i == n - 1:
            ecuacion = (electroneutralidad(vars[i], i) * 1/1+abs(i)) + (electroneutralidad(vars[i-1], i-1) * 1/1+abs(i-1))
        else:
            ecuacion = (electroneutralidad(vars[i], i) * 1/1+abs(i)) + (electroneutralidad(vars[i-1], i-1) * 1/1+abs(i-1)) + (electroneutralidad(vars[i+1], i+1) * 1/1+abs(i+1))
        
        ecuaciones.append(ecuacion)

    return ecuaciones

# Número de incógnitas
n = 100

# Valores iniciales para fsolve
valores_iniciales = np.ones(n)

# Resolver el sistema de ecuaciones
solucion = fsolve(sistema_ecuaciones, valores_iniciales)

ph_values = []

# Mostrar la solución
for i in range(n):
    ph = -np.log10(abs(solucion[i]))  # Conversión de la concentración de H+ a pH
    ph_values.append(ph)
    print(f"ph{i} = {ph_values[i]}")