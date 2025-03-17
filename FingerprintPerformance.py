import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Load your CSV file (adjust the filename and path as needed)
data = pd.read_csv('results_Viscosity_40C.csv')

# Extract the columns for plotting
x = data['Bits'].values
y = data['Radius'].values
z = data['Test_RMSE'].values  # Replace 'performance' with your actual metric column name


# Create a grid for the surface plot
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')

# Create a grid for the surface plot
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method='cubic')

# For RMSE, a lower value is better, so we use np.argmin
best_idx = np.argmin(z)
best_x = x[best_idx]
best_y = y[best_idx]
best_z = z[best_idx]

# Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with some transparency
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='none', alpha=0.8)

# Plot the best performance point with a diamond marker and include the RMSE value in the label
ax.scatter(best_x, best_y, best_z, color='red', marker='D', s=50,
           label=f'Best Performance (RMSE: {best_z:.2f})')

# Set axis labels
ax.set_xlabel('Number of Bits')
ax.set_ylabel('Radius')
ax.set_zlabel('RMSE')
ax.set_ylim(10, 0)

# Add legend and colorbar
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.title('Effect of Number of Bits and Radius on Model Performance')
plt.show()