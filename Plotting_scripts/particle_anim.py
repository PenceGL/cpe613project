import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

df = pd.read_csv('particle_data.csv')

fig, ax = plt.subplots()
# ax.set_xlim(df[['ElectronPosX', 'NearestProtonPosX']].values.min(), df[['ElectronPosX', 'NearestProtonPosX']].values.max())
# ax.set_ylim(df[['ElectronPosY', 'NearestProtonPosY']].values.min(), df[['ElectronPosY', 'NearestProtonPosY']].values.max())
ax.set_xlim([-2e-11, 2e-11])
ax.set_ylim([-2e-11, 2e-11])

# plot initialization
electrons, = ax.plot([], [], 'bo', label='Electrons', markersize=3)
protons, = ax.plot([], [], 'ro', label='Protons', markersize=3)
electron_trails = [ax.plot([], [], 'b', linewidth=1, alpha=0.5)[0] for _ in range(df['ElectronID'].nunique())]
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

# initialize a dictionary to hold the path data
paths = {eid: [] for eid in df['ElectronID'].unique()}

def init():
    electrons.set_data([], [])
    protons.set_data([], [])
    for trail in electron_trails:
        trail.set_data([], [])
    return [electrons, protons] + electron_trails

def animate(i):
    step_data = df[df['Step'] == i]
    electrons.set_data(step_data['ElectronPosX'], step_data['ElectronPosY'])
    protons.set_data(step_data['NearestProtonPosX'], step_data['NearestProtonPosY'])

    # update trails
    for eid, trail in zip(step_data['ElectronID'], electron_trails):
        path = paths[eid]
        path.append((step_data.loc[step_data['ElectronID'] == eid, 'ElectronPosX'].values[0],
                     step_data.loc[step_data['ElectronID'] == eid, 'ElectronPosY'].values[0]))
        xs, ys = zip(*path)
        trail.set_data(xs, ys)

    return [electrons, protons] + electron_trails

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=np.unique(df['Step']), interval=25, blit=False)

ani.save('particle_motion_trails.gif', writer='imagemagick')
