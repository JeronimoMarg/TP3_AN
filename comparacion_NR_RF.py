import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

def d2f(x):
    return 6*x

def newton_raphson(f, df, x0, tol=1e-7, max_iter=100):
    x_n = x0
    errors = []
    for i in range(max_iter):
        f_x_n = f(x_n)
        df_x_n = df(x_n)
        if df_x_n == 0:
            break
        x_next = x_n - f_x_n / df_x_n
        error = abs((x_next - x_n) / x_next) * 100
        errors.append(error)
        if error < tol:
            break
        x_n = x_next
    return x_n, errors

def regula_falsi(f, a, b, tol=1e-7, max_iter=100):
    x_n = a
    x_n_minus_1 = b
    errors = []
    for i in range(max_iter):
        f_x_n = f(x_n)
        f_x_n_minus_1 = f(x_n_minus_1)
        if f_x_n == f_x_n_minus_1:
            break
        x_next = x_n - f_x_n * (x_n - x_n_minus_1) / (f_x_n - f_x_n_minus_1)
        error = abs((x_next - x_n) / x_next) * 100
        errors.append(error)
        if error < tol:
            break
        x_n_minus_1 = x_n
        x_n = x_next
    return x_n, errors

def hybrid_method(f, df, d2f, a_n, b_n, x_n, x_n_minus_1, tol=1e-7, max_iter=100):
    errors = []
    for i in range(max_iter):
        f_n = f(x_n)
        f_prime_n = df(x_n)
        f_secant_slope = (f(x_n) - f(x_n_minus_1)) / (x_n - x_n_minus_1)
        
        f_double_prime_a = d2f(a_n)
        f_double_prime_b = d2f(b_n)
        
        alpha = abs(f_double_prime_a) / (abs(f_double_prime_a) + abs(f_double_prime_b))
        beta = abs(f_double_prime_b) / (abs(f_double_prime_a) + abs(f_double_prime_b))
        
        MNS = alpha * f_prime_n + beta * f_secant_slope
        
        x_next = x_n - f_n / MNS
        error = abs((x_next - x_n) / x_next) * 100
        errors.append(error)
        if error < tol:
            break
        x_n_minus_1 = x_n
        x_n = x_next
    return x_n, errors

# Valores iniciales
x0_newton = 2  # Aproximación inicial para Newton-Raphson
a_regula = 1
b_regula = 2  # Intervalo inicial para Regula-Falsi
a_n = 1
b_n = 2
x_n = b_n
x_n_minus_1 = a_n

# Ejecutar los métodos
root_newton, errors_newton = newton_raphson(f, df, x0_newton)
root_regula, errors_regula = regula_falsi(f, a_regula, b_regula)
root_hybrid, errors_hybrid = hybrid_method(f, df, d2f, a_n, b_n, x_n, x_n_minus_1)

# Mostrar los resultados
plt.plot(errors_newton, label='Newton-Raphson')
plt.plot(errors_regula, label='Regula-Falsi')
plt.plot(errors_hybrid, label='Newton-Secant Hybrid')
plt.xlabel('Iteration')
plt.ylabel('Relative Error (%)')
plt.yscale('log')
plt.legend()
plt.title('Relative Error Reduction per Iteration')
plt.show()

print(f"Root found by Newton-Raphson: {root_newton}")
print(f"Root found by Regula-Falsi: {root_regula}")
print(f"Root found by Newton-Secant Hybrid: {root_hybrid}")