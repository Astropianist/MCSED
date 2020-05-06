import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

c = 3.0e5 # km/s
wavL = 911.75
Lyn = np.arange(2,41)
bval = 35.0 #km/s
Lywav = wavL/(1.0-Lyn.astype(float)**(-2))
An = np.array([3.61e-03,1.69e-03,1.17e-3,9.27e-4,7.81e-4,6.55e-4,6.07e-4,5.49e-4,5.03e-4,4.61e-4,4.28e-4,3.99e-4,3.74e-4,3.51e-4,3.31e-4,3.12e-4,2.95e-4,2.79e-4,2.65e-4,2.51e-4,2.38e-4,2.27e-4,2.15e-4,2.05e-4,1.95e-4,1.86e-4,1.77e-4,1.69e-4,1.61e-4,1.53e-4,1.46e-4,1.39e-4,1.33e-4,1.27e-4,1.21e-4,1.16e-4,1.10e-4,1.05e-4,1.01e-4])
Narr = np.geomspace(2.0e12,1.0e20,101)
lamemarr = np.linspace(700.0,1400.0,201)

def cont(lobs,zem):
    xc = lobs/wavL
    xem = 1.0+zem
    return 0.25*xc**3*(xem**0.46-xc**0.46) + 9.4*xc**1.5*(xem**0.18-xc**0.18) - 0.7*xc**3*(xem**-1.32-xc**-1.32) - 0.023*(xem**1.68-xc**1.68)

def linesum(lobs,jmax,A=An,lam=Lywav):
    return sum(A[:jmax-1]*(lobs/lam[:jmax-1])**3.46)

def gettau(lobs,zem,lam=Lywav):
    xem = 1.0+zem
    if lobs > lam[0]*xem: return 0.0
    elif lobs > lam[-1]*xem: 
        lamem = lam*xem
        i = 0
        while lamem[i]>lobs:
            i+=1
        return linesum(lobs,i+1)
    elif lobs > wavL*xem: return linesum(lobs,len(lam)+1)
    else: return linesum(lobs,len(lam)+1) + cont(lobs,zem)

def getIGMTransp_wav(zem,wavarr):
    transp = np.zeros(len(wavarr))
    for i,lobs in enumerate(wavarr):
        transp[i] = gettau(lobs,zem)
    return transp

class IGMextinct:
    def __init__(self,z,wv):
        """ OBSERVED wv in Angstroms """
        self.z = z
        self.wv = wv

    def extinguish(self,spec):
        tau_IGM = getIGMTransp_wav(self.z,self.wv)
        return spec*np.exp(-tau_IGM)