from scipy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as Anim
import profile

def Simulation_Animation(posx,posy,velo,Vectors):
    fig, ax = plt.subplots()
    dots, = ax.plot([], [], 'ro')
    time = 0
    for j in range(0,len(posx[:,0])):
        ax.arrow(posx[j,0],posy[j,0],cos(velo[j,0])/5,sin(velo[j,0])/5)
    ax.set_xlim([0,Vectors.L])
    ax.set_ylim([0,Vectors.L])

    def init():  # only required for blitting to give a clean slate.
        line.set_xdata(posx[:,0])
        line.set_ydata(posy[:,0])
        return line, ax

    def animate(i):
        ax.clear()
        ax.set_xlim([0,Vectors.L])
        ax.set_ylim([0,Vectors.L])
        for j in range(0,len(posx[:,0])):
            ax.arrow(posx[j,i],posy[j,i],cos(velo[j,i])/5,sin(velo[j,i])/5)
        plt.title(r'Time={0}'.format(i))
        return ax

    anim = Anim.FuncAnimation(fig,animate,frames = linspace(0,Vectors.time-1,Vectors.time,dtype=int),interval=250,blit=False,repeat=True)
    
    return(anim)