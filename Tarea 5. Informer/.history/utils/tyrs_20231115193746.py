import numpy as np
import matplotlib.pyplot as plt

# Creating a range of x values from -2π to 2π
x = np.linspace(-2 * np.pi, 2 * np.pi, 400)

# Calculating the sine of each x value
y = np.cos(x)

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)

# Adding title and labels
plt.title('Gráfico de Seno')
plt.xlabel('x')
plt.ylabel('sin(x)')

# Display the plot
plt.grid(True)
plt.show()
