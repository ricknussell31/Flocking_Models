# coding: utf-8

from math import acos, asin
import numpy as np
from numpy.random import rand, uniform

class Background_Field(object):
    
    # class builder initiation 
    def __init__(self,L=5,N=300,eta=.02,r=1,dt=1,time=50,*args,**kwargs):
        self.L = L #length of LxL domain
        self.N = N #number of particles
        self.eta = eta #noise for direction
        self.r = r #interaction radius
        self.dt = dt #timestep
        self.time = time #total iterations
        
    # Set up initial condition
    def SetIC(self):
        posx = np.zeros((self.N,self.time))
        posy = np.zeros((self.N,self.time))
        velo = np.zeros((self.N,self.time))

        posx[:,0] = self.L*rand(self.N) #initial positions x-coordinates
        posy[:,0] = self.L*rand(self.N) #initial poisitions y-coordinates
        velo[:,0] = 2*np.pi*rand(self.N) #initial velocities
        return(posx,posy,velo)
    
    def ThetaNoise(self):
        #This is a uniform disribution of noise
        return(uniform(-self.eta/2,self.eta/2,self.N))
    
class Particles(Background_Field):
    
    def __init__(self,v=0.3, ep=1, k = 0,*args,**kwargs):

        self.v = v #constant speed
        self.ep = ep #the delay parameter (ep = 1 implies the Vicsek model, ep = 0 is ballistic movement)
        
        #number of nearest neighbors or case number. If k = 0, this is the metric model. For k > 0, we use the top. model.
        if (type(k) == int):
            self.k = int(k)
        else:
            self.k = int(k)
            print('You did not provide an integer. To correct this, we have made your value of k = {0}'.format(self.k))
        self.args = args
        self.kwargs = kwargs
        
        super(Particles,self).__init__(*args,**kwargs)
    
    def Simulation(self):
        #This function simulates the scheme selected by the user for the flocking mechanism
        #k = 0 represents the metric case where k > 0 represents the topological case with k nearest neighbors
        
        #Inputs -- None
        #Outputs -- Positions and velocities of all timesteps of the simulation
        posx, posy, velo = self.SetIC()
        if (self.k == 0):
            for d in range(1,self.time):
                posx[:,d],posy[:,d],velo[:,d] = self.Update_Metric(posx[:,d-1],posy[:,d-1],velo[:,d-1])
        elif (self.k > 0):
            for d in range(1,self.time):
                posx[:,d],posy[:,d],velo[:,d] = self.Update_Top(posx[:,d-1],posy[:,d-1],velo[:,d-1],self.k)
        elif (self.k < 0):
            print('k must be positive. Please reinitialize with a proper k')
        return(posx,posy,velo)
    
    def Update_Metric(self,posx,posy,velo):
        #This method allows us to update the flocking model with the Metric model
        #Each new velocity is constructed by averaging over all of the velocities within
        #the radius selected, self.r.
        #Inputs -- x-coordinates, y-coordinates, trajectories for time = t
        #Outputs -- x-coordinates, y-coordinates, trajectories for time = t + (delta t)
        avgs = 0*velo
        avgc = 0*velo
        
        for j in range(0,self.N):
            #find distances for all particles
            Dist = self.Calc_Dist(posx,posy,self.L,j)
            
            #find indicies that are within the radius
            Vals = [i for i in range(len(Dist)) if Dist[i] <= self.r]
            
            #find average velocity of those inside the radius
            sint = 0 
            cost = 0
            for k in Vals:
                sint = sint + np.sin(velo[k])
                cost = cost + np.cos(velo[k])
            if (len(Vals)==0):
                #catch yourself if there are no objects within the desired radius
                avgs[j] = 0
                avgc[j] = 0
            else:
                avgs[j] = sint/len(Vals)
                avgc[j] = cost/len(Vals)
                
        #construct the noise
        noise = self.ThetaNoise()
        
        #update velocities and positions
        cosi = (self.ep)*avgc+(1-self.ep)*np.cos(velo)
        sini = (self.ep)*avgs+(1-self.ep)*np.sin(velo)
        newvelo = np.arctan2(sini,cosi) 
        velon = np.mod(newvelo + noise,2*np.pi)
        posx = posx + self.dt*self.v*np.cos(velon) 
        posy = posy + self.dt*self.v*np.sin(velon)
        
        #Make sure that the positions are not outside the boundary.
        #If so, correct for periodicity
        posx,posy = self.CheckBoundary(posx,posy)
        
        #Outputs returned
        return(posx,posy,velon)
    
    def Update_Top(self,posx,posy,velo,kn):
        #This is the topological mdoel for flocking. This finds the k-nearest neighbors 
        #k here is the number of nearest neighbors selected to average over
        avgs = 0*velo
        avgc = 0*velo
        L = self.L
        for j in range(0,self.N):
            
            #calculate distances
            Dist = self.Calc_Dist(posx,posy,self.L,j)
            
            #Find k-nearest neighbors
            Vals = Dist.argsort()[:kn] #indices of the k-nearest neighbors

            #find average velocity
            sint = 0 
            cost = 0
            for k in Vals:
                sint = sint + np.sin(velo[k])
                cost = cost + np.cos(velo[k])
            avgs[j] = sint/len(Vals)
            avgc[j] = cost/len(Vals)

        noise = self.ThetaNoise()
        
        #update velocities and positions
    
        cosi = (self.ep)*avgc+(1-self.ep)*np.cos(velo)
        sini = (self.ep)*avgs+(1-self.ep)*np.sin(velo)
        newvelo = np.arctan2(sini,cosi)
        velon = np.mod(newvelo + noise,2*np.pi)
        posx = posx + self.dt*self.v*np.cos(velon) 
        posy = posy + self.dt*self.v*np.sin(velon)
        
        #make sure individuals are not outside the boundary, and if they are
        #utilize he periodicity to get them in the right place.
        posx,posy = self.CheckBoundary(posx,posy)
        
        #output information for next timesteps
        return(posx,posy,velon)
    
    
    def Calc_Dist(self,posx,posy,L,j):
        #find distance of every particle from particle j using periodic boundary conditions
        
        Dist0 = np.sqrt((posx[j] - posx)**2 + (posy[j] - posy)**2) #regular  
        Dist1 = np.sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy + L)**2) #topleft
        Dist2 = np.sqrt((posx[j]  - posx)**2 + (posy[j] - posy + L)**2) #topcenter
        Dist3 = np.sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy + L)**2) #topright
        Dist4 = np.sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy)**2) #middleleft
        Dist5 = np.sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy)**2) #middleright
        Dist6 = np.sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy - L)**2) #bottomleft
        Dist7 = np.sqrt((posx[j]  - posx)**2 + (posy[j] - posy - L)**2) #bottomcenter
        Dist8 = np.sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy - L)**2) #bottomright
        
        TD = [Dist0,Dist1,Dist2,Dist3,Dist4,Dist5,Dist6,Dist7,Dist8]
        
        return(np.asarray(TD).min(0)) #minimum values for all possible distances
    
    def CheckBoundary(self,posx,posy):
        xcordn = [i for i in range(self.N) if posx[i] < 0]
        xcordp = [i for i in range(self.N) if posx[i] > self.L]
        ycordn = [i for i in range(self.N) if posy[i] < 0]
        ycordp = [i for i in range(self.N) if posy[i] > self.L]
        
        for j in xcordn:
            posx[j] = posx[j] + self.L
       
        for j in xcordp:
            posx[j] = posx[j] - self.L
            
        for j in ycordn:
            posy[j] = posy[j] + self.L
            
        for j in ycordp:
            posy[j] = posy[j] - self.L
           
        return(posx,posy)
                                      