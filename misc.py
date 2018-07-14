import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import matplotlib.pyplot as plt


def plot_flight(flight_path, target_pos, episode=None, fig = None):
    x = flight_path[0]
    y = flight_path[1]
    z = flight_path[2]
    mpl.rcParams['legend.fontsize'] = 15

    if fig == None:
        fig = plt.figure()
        fig.set_size_inches(12, 12)
    ax = fig.gca(projection='3d')

    ax.plot(x,y,z, label='Flight of the Quadcopter')
    ax.plot([target_pos[0]], [target_pos[1]], [target_pos[2]], ms=8, lw=2, marker='x', alpha=0.75, label='Target')
    ax.plot([x[0]],[y[0]],[z[0]] ,    ms=8, lw=2, marker='o', color='g', label='Start')
    ax.plot([x[-1]],[y[-1]],[z[-1]] , ms=8, lw=2, marker='x', color='g', label='End')
    #ax.legend()
    xmax = np.max(x)
    xmin = np.min(x)
    ymax = np.max(y)
    ymin = np.min(y)
    zmax = np.max(z)
    zmin = np.min(z)
    #zmax = np.max(z)
    plt_buffer = 2
    pltmax = math.ceil(max(xmax,ymax) + plt_buffer)
    ax.set_xlim(-pltmax,pltmax)
    ax.set_ylim(-pltmax,pltmax)
    #ax.set_zlim(0,pltmax)
    if episode != None:
        title_txt = 'Episode ' + str(episode)
    else:
        title_txt = 'No Episode'
    if z[0] > 0:
        title_txt += '. Flight starts in mid-air at z=' + str(np.round(z[0])) + '. Range Max is ' + str(pltmax)
    plt.title(title_txt)
    ax.set_xlabel('X (' + str(round(xmin,1)) + ', ' + str(round(xmax,1)) + ')')
    ax.set_ylabel('Y (' + str(round(ymin,1)) + ', ' + str(round(ymax,1)) + ')')
    ax.set_zlabel('Z (' + str(round(zmin,1)) + ', ' + str(round(zmax,1)) + ')')
    plt.show()