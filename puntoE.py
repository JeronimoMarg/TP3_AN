import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constantes
pKa_acetico = 4.756
pKa_amoniaco = 9.25

Ka_acetico = 10**(-pKa_acetico)
Kb_amoniaco = 10**(-pKa_amoniaco)

Kw = 10**(-14)  # Constante de disociación del agua

N = 100  # Número de puntos a lo largo de la columna

# Definir las concentraciones lineales
conc_acido = np.linspace(1e-3, 10e-3, N)  # 1 mM a 10 mM
conc_amoniaco = np.linspace(10e-3, 1e-3, N)  # 10 mM a 1 mM

def electroneutralidad(H, i):
    NH4 = (conc_amoniaco[i] * H) / (Kb_amoniaco + H)
    CH3COO = conc_acido[i] / ((Ka_acetico * H) + 1)
    OH = Kw / H
    return H - OH + NH4 - CH3COO

def promedio_ponderado(H, i, j):
    suma = 0
    for k in range(-j, j+1):
        if 0 <= i+k < len(H):
            suma += electroneutralidad(H[i+k], i+k) / (1 + abs(k))
    return suma

def sistema_ecuaciones(vars, j):
    n = len(vars)
    ecuaciones = np.zeros(n)
    for i in range(n):
        ecuaciones[i] = promedio_ponderado(vars, i, j)
    return ecuaciones

# Calcular el Jacobiano de forma numérica
def calcular_jacobiano(func, x, j):
    n = len(x)
    jacobiano = np.zeros((n, n))
    h = 1e-8
    fx = func(x, j)
    
    for i in range(n):
        x_h = np.copy(x)
        x_h[i] += h
        jacobiano[:, i] = (func(x_h, j) - fx) / h
    
    return jacobiano

# Resolver el sistema de ecuaciones
initial_guess = np.ones(N) * 1e-7  # Supongamos un pH inicial de 7

'''
# Método directo usando fsolve y cálculo numérico del Jacobiano
solution_directa = fsolve(lambda vars: sistema_ecuaciones(vars, j=1), initial_guess, fprime=lambda vars: calcular_jacobiano(sistema_ecuaciones, vars, j=1))

# Calcular el pH a partir de las concentraciones de H+
concentraciones_H_directa = np.where(solution_directa > 0, solution_directa, 1e-20)
pH_directa = -np.log10(concentraciones_H_directa)

print("Solución método directo:")
print(solution_directa)
'''

# Método iterativo de Jacobi
def metodo_jacobi(j, max_iter=1000, tol=1e-10):
    A = calcular_jacobiano(sistema_ecuaciones, initial_guess, j)
    b = -sistema_ecuaciones(initial_guess, j)
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    for i in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    #x = np.where(x > 0, x, 1e-20)  # Asegurar valores positivos
    return x

# Método iterativo de Gauss-Seidel
def metodo_gauss_seidel(j, max_iter=1000, tol=1e-10):
    A = calcular_jacobiano(sistema_ecuaciones, initial_guess, j)
    b = -sistema_ecuaciones(initial_guess, j)
    x = np.zeros_like(b)
    
    for i in range(max_iter):
        x_new = np.copy(x)
        for k in range(A.shape[0]):
            x_new[k] = (b[k] - np.dot(A[k, :k], x_new[:k]) - np.dot(A[k, k+1:], x[k+1:])) / A[k, k]
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    #x = np.where(x > 0, x, 1e-20)  # Asegurar valores positivos
    return x

# Método LU Decomposition
from scipy.linalg import lu, solve

def metodo_lu_decomposition(j):
    A = calcular_jacobiano(sistema_ecuaciones, initial_guess, j)
    b = -sistema_ecuaciones(initial_guess, j)
    
    P, L, U = lu(A)
    y = solve(L, np.dot(P.T, b))
    x = solve(U, y)
    
    #x = np.where(x > 0, x, 1e-20)  # Asegurar valores positivos
    return x

# Comparar los métodos
sol_jacobi = metodo_jacobi(j=1)
sol_gauss_seidel = metodo_gauss_seidel(j=1)
sol_lu = metodo_lu_decomposition(j=1)

print("Solución método Jacobi:")
print(sol_jacobi)

print("Solución método Gauss-Seidel:")
print(sol_gauss_seidel)

print("Solución método LU Decomposition:")
print(sol_lu)

# Calcular el pH a partir de las concentraciones de H+
concentraciones_H_jacobi = np.where(sol_jacobi > 0, sol_jacobi, 1e-20)
concentraciones_H_gauss_seidel = np.where(sol_gauss_seidel > 0, sol_gauss_seidel, 1e-20)
concentraciones_H_lu = np.where(sol_lu > 0, sol_lu, 1e-20)

pH_jacobi = -np.log10(concentraciones_H_jacobi)
pH_gauss_seidel = -np.log10(concentraciones_H_gauss_seidel)
pH_lu = -np.log10(concentraciones_H_lu)

# Graficar el pH para los diferentes métodos
#plt.plot(range(N), pH_directa, marker='o', label='Directo')
plt.plot(range(N), pH_jacobi, marker='x', label='Jacobi')
plt.plot(range(N), pH_gauss_seidel, marker='+', label='Gauss-Seidel')
plt.plot(range(N), pH_lu, marker='*', label='LU Decomposition')
plt.xlabel('Índice')
plt.ylabel('pH')
plt.title('Comparación de pH en función de las concentraciones de H+')
plt.legend()
plt.grid(True)
plt.show()
