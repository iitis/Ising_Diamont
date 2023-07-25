import numpy as np
from scipy import *

def w(s:float, mu:int, beta:float, J:float, h0:float, Jz:float, h:float, Jm:float, Jp:float):
    if mu==1:
        g=J/4 + h0/2
        l=J+h
    elif mu==-1:
        g=-J/4 -h0/2
        l=-J+h
    elif mu==0:
        g=-J/4
        l=h
    if s==0.5:
        x=2*np.exp(-beta*(g + Jz/4))*np.cosh(beta*np.sqrt(16*l**2 + Jm**2)/4) + 2*np.exp(-beta*(g - Jz/4))*np.cosh(beta*Jp/4)
    elif s==1.0:
        x=np.exp(-beta*g)*(np.exp(beta*Jz)+2*np.exp(beta*Jz/2)*np.cosh(beta*np.sqrt(Jz**2 + 2*Jp**2)/2)+4*np.cosh(beta*Jp/2)*np.cosh(beta*l)+
                                2*np.exp(-beta*Jz)*np.cosh(beta*2*l))
    return x
    

s=1.0, mu=1, beta=1.0, J=1.0, h0=0.5, Jz=1.0, h=0.3, Jm=0.6, Jp=5.0

w(s,mu,beta,J,h0,Jz,h,Jm,Jp)