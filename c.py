import numpy as np
import matplotlib.pyplot as plt

def newton_secante(f, df, d2f, xn, xn_1, tol=1e-7, max_iter=100):
    
    errors = []

    for i in range(max_iter):
        fxn = f(xn)
        fxn_1 = f(xn_1)
        pendienteSecante = (fxn_1 - fxn) / (xn_1 - xn)
        pendienteTangente = df(xn)
        
        derivSegundaA = d2f(xn_1)
        derivSegundaB = d2f(xn)
        
        alpha = abs(derivSegundaA) / (abs(derivSegundaA) + abs(derivSegundaB))
        beta = abs(derivSegundaB) / (abs(derivSegundaA) + abs(derivSegundaB))
        
        MNS = alpha * pendienteTangente + beta * pendienteSecante
        
        xNext = xn - (fxn / MNS)
        error = abs((xNext - xn) / xNext) * 100
        errors.append(error)
        if error < tol:
            break
        xn_1 = xn
        xn = xNext

    return abs(xn), errors


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


# Calcular el pH para cada punto
for i in range(N):
    H, errors = newton_secante(
        lambda H: electroneutralidad(H, i),
        lambda H: electroneutralidad_primera_derivada(H, i),
        lambda H: electroneutralidad_segunda_derivada(H, i),
        initial_1,
        initial_2
    )
    if H > 0:
        ph = -np.log10(H)  # Conversión de la concentración de H+ a pH
        ph_values.append(ph)
        print(f"pH en punto {i+1}: {ph:.5f} en {len(errors)} iteraciones.")
        print(f"H: {H}")
    else:
        ph_values.append(np.nan)
        print(f"pH en punto {i+1}: no valido (H = {H})")
        
        
# Graficar resultados
plt.plot(range(1, N+1), ph_values, marker='o', linestyle='-')
plt.xlabel('Punto a lo largo de la columna')
plt.ylabel('pH')
plt.title('Variación del pH a lo largo de la columna de separación')
plt.grid(True)
plt.show()



