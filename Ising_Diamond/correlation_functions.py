import numpy as np
import numdifftools as nd
import findiff
from scipy import *
import matplotlib.pylab as plt

## Boltzmann weights according to spin-s (Eq. 51 - Valverde et al 2008)
def w(s:str, mu:int, beta:float, J:float, h0:float, Jz:float, h:float, Jm:float, Jp:float):
    if mu==1:
        g=J/4 + h0/2
        l=J+h
    elif mu==-1:
        g=-J/4 -h0/2
        l=-J+h
    elif mu==0:
        g=-J/4
        l=h
    if s=='1/2':
        x=2*np.exp(-beta*(g + Jz/4))*np.cosh(beta*np.sqrt(16*l**2 + Jm**2)/4) + 2*np.exp(-beta*(g - Jz/4))*np.cosh(beta*Jp/4)
    elif s=='1':
        x=np.exp(-beta*g)*(np.exp(beta*Jz)+2*np.exp(beta*Jz/2)*np.cosh(beta*np.sqrt(Jz**2 + 2*Jp**2)/2)+4*np.cosh(beta*Jp/2)*np.cosh(beta*l)+
                                2*np.exp(-beta*Jz)*np.cosh(beta*2*l))
    return x
    
## magnetization <spin-s> (Eq. 30 - Carvalho et al 2019)
def mag(beta:float, J:float, h0:float, Jz:float, h:float, Jm:float, Jp:float):
    s='1/2'
    w0_tilda=w(s,0,beta,J,h0,Jz,h,Jm,Jp)/abs(w(s,1,beta,J,h0,Jz,h,Jm,Jp)-w(s,-1,beta,J,h0,Jz,h,Jm,Jp))
    m=(1/2) * ((w(s,1,beta,J,h0,Jz,h,Jm,Jp)-w(s,-1,beta,J,h0,Jz,h,Jm,Jp))/abs(w(s,1,beta,J,h0,Jz,h,Jm,Jp)-w(s,-1,beta,J,h0,Jz,h,Jm,Jp))) * (1/np.sqrt(1+4*w0_tilda**2))
    return m

## Pauli spin-spin correlation functions (Eq. 34 - Carvalho et al 2019)
def sigma_i_sigma_j(beta:float,r:int, J:float, h0:float, Jz:float, h:float, Jm:float, Jp:float):
    s='1/2'
    B=np.sqrt((w(s,1,beta,J,h0,Jz,h,Jm,Jp)-w(s,-1,beta,J,h0,Jz,h,Jm,Jp))**2 + 4*w(s,0,beta,J,h0,Jz,h,Jm,Jp)**2)
    sisj=mag(beta,J,h0,Jz,h,Jm,Jp)**2 + (w(s,0,beta,J,h0,Jz,h,Jm,Jp)/B)**2 * ((w(s,1,beta,J,h0,Jz,h,Jm,Jp)+w(s,-1,beta,J,h0,Jz,h,Jm,Jp)-B)/(w(s,1,beta,J,h0,Jz,h,Jm,Jp)+w(s,-1,beta,J,h0,Jz,h,Jm,Jp)+B))**r
    return sisj

## Heisenberg spin-spin correlation function in the z direction (Eq. 34 - Carvalho et al 2019)

def wz_mu(mu:int, beta:float, J:float, h0:float, Jz:float, h:float, Jm:float, Jp:float):
    s='1'
    wz=(1/(2*beta)) * w(s,mu, beta, J, h0, Jz, h, Jm, Jp)
    df=nd.Derivative(wz,n=1)
    return df(h)

def Sz(beta:float, J:float, h0:float, Jz:float, h:float, Jm:float, Jp:float):
    s='1'
    B=np.sqrt((w(s,1,beta,J,h0,Jz,h,Jm,Jp)-w(s,-1,beta,J,h0,Jz,h,Jm,Jp))**2 + 4*w(s,0,beta,J,h0,Jz,h,Jm,Jp)**2)
    lp=(w(s,1,beta,J,h0,Jz,h,Jm,Jp)+w(s,-1,beta,J,h0,Jz,h,Jm,Jp)+B)/2
    return sz

s='1'; mu=1; beta=1; J=1.0; h0=0.5; Jz=1.0; Jm=0.6; Jp=5.0
h=2
data=lambda h: w(s,mu,beta,J,h0,Jz,h,Jm,Jp)
df=nd.Derivative(data,n=1)
y = df(h)
print(y/abs(y))