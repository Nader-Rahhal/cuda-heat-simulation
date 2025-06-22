import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def read_temperature_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [list(map(float, line.strip().split())) for line in lines]
    return np.array(data)

def plot_heat_surface(data):
    time_steps, rod_length = data.shape
    x = np.arange(rod_length)  # Position on rod
    z = np.arange(time_steps)  # Time steps
    X, Z = np.meshgrid(x, z)
    Y = data  # Temperature

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Z, Y, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('Rod Position (X)')
    ax.set_ylabel('Time Step (Z)')
    ax.set_zlabel('Temperature (Y)')
    ax.set_title('1D Heat Diffusion Over Time')

    fig.colorbar(surf, shrink=0.5, aspect=10, label='Temperature')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = read_temperature_data("output_1d.txt")
    plot_heat_surface(data)

