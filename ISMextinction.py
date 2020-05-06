from dust_extinction.parameter_averages import F99
import numpy as np
from scipy.interpolate import interp1d

class ISMextinct:
    def __init__(self,wv,ebv,Rv=3.1):
        """ OBSERVED wv in Angstroms, Rv is Av/E(B-V)"""
        self.wv = wv
        self.Rv = Rv
        self.ebv = ebv
        self.F99obj = F99(Rv=self.Rv)

    def evaluate(self):
        Rlam = np.zeros(len(self.wv))
        cond = np.logical_and(self.wv<3.33e4,self.wv>1000.0)
        wvnum = 1.0e4/self.wv #Wavenumber in micron^-1
        Rlam[cond] = self.F99obj(wvnum[cond])
        Rlamext = interp1d(self.wv[cond],Rlam[cond],kind='linear',fill_value='extrapolate')
        Rlam[self.wv>3.33e4] = 0.0
        Rlam[self.wv<1000.0] = Rlamext(self.wv[self.wv<1000.0])
        return Rlam

    def get_tau(self,Rlam=None):
        if Rlam is None: Rlam=self.evaluate()
        Alam = Rlam*self.ebv
        return Alam/1.086

    def extinguish(self,spec):
        Rlam = self.evaluate()
        tau_ISM = self.get_tau(Rlam)
        return spec*np.exp(-tau_ISM)