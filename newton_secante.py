def f(x):
    return x**3 - 2*x

def df(x):
    return 3*x**2 - 2

def d2f(x):
    return 6*x

def hybrid_method(f, df, d2f, a_n, b_n, x_n, x_n_minus_1):
    f_n = f(x_n)
    f_prime_n = df(x_n)
    f_secant_slope = (f(x_n) - f(x_n_minus_1)) / (x_n - x_n_minus_1)
    
    f_double_prime_a = d2f(x_n_minus_1)
    f_double_prime_b = d2f(x_n)

    '''
    f_double_prime_a = d2f(a_n)
    f_double_prime_b = d2f(b_n)
    '''
    
    alpha = abs(f_double_prime_a) / (abs(f_double_prime_a) + abs(f_double_prime_b))
    beta = abs(f_double_prime_b) / (abs(f_double_prime_a) + abs(f_double_prime_b))
    
    MNS = alpha * f_prime_n + beta * f_secant_slope
    print("La pendiente es: ", MNS)
    
    x_next = x_n - f_n / MNS
    return x_next

# Parâmetros de inicio
a_n = 0.2
b_n = 2.6
x_n = b_n
x_n_minus_1 = a_n
tolerance = 1e-7
max_iterations = 100
iteration = 0

# Iteração
while abs(f(x_n)) > tolerance and iteration < max_iterations:
    x_next = hybrid_method(f, df, d2f, a_n, b_n, x_n, x_n_minus_1)
    x_n_minus_1 = x_n
    x_n = x_next
    iteration += 1
    print(f"Iteration {iteration}: x_n = {x_n}, f(x_n) = {f(x_n)}")

# Resultado final
if abs(f(x_n)) <= tolerance:
    print(f"Root found: x = {x_n}, f(x) = {f(x_n)}")
else:
    print("Maximum iterations reached without finding a root.")