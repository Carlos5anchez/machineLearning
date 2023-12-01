#Carlos Sanchez
import random
import matplotlib.pyplot as plt
import numpy as np

# Definición de las funciones
def f(x):
    return x**2

def g(x):
    return np.exp(-x**2/2)

# Método de rechazo para generar puntos aleatorios
def rejection_sampling(func, x_range, max_value, n_samples=1000):
    samples = []
    
    while len(samples) < n_samples:
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(0, max_value)
        
        if y < func(x):
            samples.append((x, y))
    
    return samples

# Parámetros
x_range = (0, 2)
max_value_f = f(2)
max_value_g = 1  # Para g(x) en x=0
n_samples = 1000

# Generar puntos aleatorios
samples_f = rejection_sampling(f, x_range, max_value_f, n_samples)
samples_g = rejection_sampling(g, x_range, max_value_g, n_samples)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot para f(x) = x^2
x_vals = np.linspace(x_range[0], x_range[1], 400)
axs[0].plot(x_vals, [f(x) for x in x_vals], 'r', label='f(x) = x^2')
axs[0].scatter([s[0] for s in samples_f], [s[1] for s in samples_f], s=5, color='blue')
axs[0].set_title('Muestra aleatoria para f(x) = x^2')
axs[0].legend()

# Plot para g(x) = e^{-x^2/2}
axs[1].plot(x_vals, [g(x) for x in x_vals], 'r', label='g(x) = e^{-x^2/2}')
axs[1].scatter([s[0] for s in samples_g], [s[1] for s in samples_g], s=5, color='blue')
axs[1].set_title('Muestra aleatoria para g(x) = e^{-x^2/2}')
axs[1].legend()

plt.tight_layout()
plt.show()
