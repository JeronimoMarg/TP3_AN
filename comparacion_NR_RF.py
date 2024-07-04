import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1/4 * (x**3 - 2*x)

def df(x):
    return 3/4*(x**2) - 1/2

def d2f(x):
    return 3/2*x

def newton_raphson(f, df, x0, tol=1e-7, max_iter=100):
    '''
    Implementación del método de Newton para encontrar la raíz de una función.

    Parámetros:
    f (función): La función de la cual queremos encontrar la raíz.
    df (función): La derivada de la función f.
    x0 (float): Valor inicial de la aproximación.
    tol (float): Tolerancia para el criterio de convergencia.
    max_iter (int): Número máximo de iteraciones.

    Retorna:
    (float): Aproximación de la raíz.
    []: Errores porcentuales por iteracion.
    '''
    errores = []
    x=x0

    for i in range (max_iter):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise ValueError("La derivada es cero. No se puede seguir con Newton")
        
        xNext = x - (fx / dfx)

        error_relativo = abs((xNext - x) / xNext)
        error_porcentual = error_relativo * 100
        errores.append(error_porcentual)
        if (error_relativo < tol):
            break
    
        x = xNext
    
    return x, errores

def regula_falsi(f, a, b, tol=1e-7, max_iter=100):

    '''
    Implementación del método de Regula Falsi para encontrar la raíz de una función en un intervalo [a, b].

    Parámetros:
    f (función): La función de la cual queremos encontrar la raíz.
    a (float): Extremo izquierdo del intervalo.
    b (float): Extremo derecho del intervalo.
    tol (float): Tolerancia para el criterio de convergencia.
    max_iter (int): Número máximo de iteraciones.

    Retorna:
    (float): Aproximación de la raíz.
    []: Errores porcentuales por iteracion.
    '''

    errores = []
    fa= f(a)
    fb=f(b)

    if(fa*fb >= 0):
        raise ValueError("La funcion debe cambiar de signo en el intervalo [a;b]")

    for i in range(max_iter):
        '''c = b - (fb * (b-a)) / (fb - fa)'''
        c = ((a * fb) - (b * fa)) / (fb - fa)

        fc = f(c)

        if (fa * fc < 0):
            b = c
        else:
            a = c

        if(i>0):
            error_relativo = abs((c - c_prev) / c)
            error_porcentual = error_relativo * 100
            errores.append(error_porcentual)
            if (error_relativo < tol):
                break
            
        c_prev = c

    return c, errores


def newton_secante(f, df, d2f, xn, xn_1, tol=1e-7, max_iter=100):

    '''
    Implementación del método de Newton-Secante para encontrar la raíz de una función en un intervalo [a, b].

    Parámetros:
    f (función): La función de la cual queremos encontrar la raíz.
    df (funcion): La derivada de f
    d2f (funcion): La derivada de d2f
    xn (float): extremo derecho del intervalo
    xn_1 (float): extremo izquierdo del intervalo
    tol (float): Tolerancia para el criterio de convergencia.
    max_iter (int): Número máximo de iteraciones.

    Retorna:
    (float): Aproximación de la raíz.
    []: Errores porcentuales por iteracion.
    '''
        
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

    return xn, errors

#VALORES PARA NEWTON
x0_newton = 2

#VALORES PARA REGULA-FALSI
a_regula = 1
b_regula = 2

#VALORES PARA NEWTON-SECANTE
a_n = 1
b_n = 2

# Ejecutar los métodos
root_newton, errors_newton = newton_raphson(f, df, x0_newton)
root_regula, errors_regula = regula_falsi(f, a_regula, b_regula)
root_hybrid, errors_hybrid = newton_secante(f, df, d2f, a_n, b_n)

# Mostrar los resultados
plt.plot(errors_newton, label='Newton-Raphson')
plt.plot(errors_regula, label='Regula-Falsi')
plt.plot(errors_hybrid, label='Newton-Secante')
plt.xlabel('Iteracion')
plt.ylabel('Error relativo (%)')
plt.yscale('log')
plt.legend()
plt.title('Error relativo % por iteracion')
plt.show()

print(f"Root found by Newton-Raphson: {root_newton} en {len(errors_newton)} iteraciones.")
print(f"Root found by Regula-Falsi: {root_regula} en {len(errors_regula)} iteraciones.")
print(f"Root found by Newton-Secant Hybrid: {root_hybrid} en {len(errors_hybrid)} iteraciones.")