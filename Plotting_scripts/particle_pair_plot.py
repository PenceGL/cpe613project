import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data = pd.read_csv('particle_data.csv')

step = data['Step']
distance = data['Distance']
ePosX = data['ElectronPosX']
ePosY = data['ElectronPosY']

plt.figure(figsize=(12, 12))

# interpolation functions
f_distance = interp1d(step, distance, kind='cubic')
f_ePosX = interp1d(step, ePosX, kind='cubic')
f_ePosY = interp1d(step, ePosY, kind='cubic')

# new interpolated steps
step_new = np.linspace(step.min(), step.max(), 500)

# plot the interpolated data
plt.plot(step_new, f_distance(step_new), linestyle='-', linewidth=2, label='ProtonDistance')
plt.plot(step_new, f_ePosX(step_new), linestyle='-', linewidth=1, label='ElectronPosX')
plt.plot(step_new, f_ePosY(step_new), linestyle='-', linewidth=1, label='ElectronPosY')

min_ePosX = np.min(ePosX)
max_ePosX = np.max(ePosX)

min_ePosY = np.min(ePosY)
max_ePosY = np.max(ePosY)

# add horizontal lines for minima and maxima
plt.axhline(y=min_ePosX, color='green', linestyle='--', linewidth=1, label='Min ePosX')
plt.axhline(y=max_ePosX, color='green', linestyle='--', linewidth=1, label='Max ePosX')
plt.axhline(y=min_ePosY, color='blue', linestyle='--', linewidth=1, label='Min ePosY')
plt.axhline(y=max_ePosY, color='blue', linestyle='--', linewidth=1, label='Max ePosY')

constant_value = 5.29177210903e-11  # Bohr radius of a Hydrogen atom
plt.axhline(y=constant_value, color='red', linestyle='-', linewidth=2, label='H Bohr radius (~5.3e-11)')

plt.xlabel('Time Step')
plt.ylabel('Distance (m)')
plt.title('Distance between Electron and Proton over Time')
plt.yscale('linear')

# plt.xlim(0, 1000)
# plt.ylim(0, 1e-9)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig('particle_plot_smoothed.png')