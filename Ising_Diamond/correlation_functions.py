import numpy as np
from scipy import *
import matplotlib.pylab as plt
import numdifftools as nd

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

## Free energy: log of the largest eigenvalue of the transfert matrix (thermodynamic limit) 
def F(s:str, beta:float, J:float, h0:float, Jz:float, h:float, Jm:float, Jp:float):
    B=np.sqrt((w(s,1,beta,J,h0,Jz,h,Jm,Jp)-w(s,-1,beta,J,h0,Jz,h,Jm,Jp))**2 + 4*w(s,0,beta,J,h0,Jz,h,Jm,Jp)**2)
    lp=(w(s,1,beta,J,h0,Jz,h,Jm,Jp)+w(s,-1,beta,J,h0,Jz,h,Jm,Jp)+B)/2
    free_energy=(-1/beta) * np.log(lp)
    return free_energy
    
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


def Sz(s:str,beta:float, J:float, h0:float, Jz:float, h:float, Jm:float, Jp:float):
    a=lambda h:F(s,beta,J,h0,Jz,h,Jm,Jp)
    da=nd.Derivative(a,n=1)
    df=da(h)
    sz=df/abs(df).max()
    return sz

def SzSz(beta:float, J:float, h0:float, Jz:float, h:float, Jm:float, Jp:float):
    a=lambda Jz : F('1',beta,J,h0,Jz,h,Jm,Jp)
    da=nd.Derivative(a,n=1)
    df=da(Jz)
    szsz=df/abs(df).max()
    return szsz

def SxSx(beta:float, J:float, h0:float, Jz:float, h:float, Jp:float):
    a=lambda Jp : F('1',beta,J,h0,Jz,h,0,Jp)
    da=nd.Derivative(a,n=1)
    df=da(Jp)
    sxsx=2 * df/abs(df).max()
    return sxsx

def Shsi(s,beta,J,h0,Jz,h,Jm,Jp):
    df= nd.Derivative(lambda h : np.log(w(s,0,beta,J,h0,Jz,h,Jm,Jp) * np.sqrt(w(s,1,beta,J,h0,Jz,h,Jm,Jp) * w(s,-1,beta,J,h0,Jz,h,Jm,Jp))),n=1)
    q0=-(1/(2*beta)) * df(h)#/abs(df(h)).max()
    dB=nd.Derivative( lambda h: -(1/(2*beta)) * np.log(w(s,1,beta,J,h0,Jz,h,Jm,Jp)/w(s,-1,beta,J,h0,Jz,h,Jm,Jp)) ,n=1)
    q10=(1/2) * dB(h)#/abs(dB(h)).max()
    dK=nd.Derivative( lambda h: -(4/beta) * np.log(w(s,1,beta,J,h0,Jz,h,Jm,Jp)*w(s,-1,beta,J,h0,Jz,h,Jm,Jp)/(w(s,0,beta,J,h0,Jz,h,Jm,Jp)**2)) ,n=1)
    q11=(1/8) * dK(h)#/abs(dK(h)).max()

    B=np.sqrt((w(s,1,beta,J,h0,Jz,h,Jm,Jp)-w(s,-1,beta,J,h0,Jz,h,Jm,Jp))**2 + 4*w(s,0,beta,J,h0,Jz,h,Jm,Jp)**2)
    l=(w(s,1,beta,J,h0,Jz,h,Jm,Jp)+w(s,-1,beta,J,h0,Jz,h,Jm,Jp)+B)/2
    corr=np.log(l) ## this is 1/correlation length
    mixed_spin=(q0*mag(beta,J,h0,Jz,h,Jm,Jp) + 2 * q10 * (mag(beta,J,h0,Jz,h,Jm,Jp)**2 + (1 - mag(beta,J,h0,Jz,h,Jm,Jp)**2) * np.exp(-corr) ) +
                q11 *  ( mag(beta,J,h0,Jz,h,Jm,Jp)**3 + mag(beta,J,h0,Jz,h,Jm,Jp) * (1 - mag(beta,J,h0,Jz,h,Jm,Jp)**2) * (1+2*np.exp(-corr)) ) )
    return mixed_spin
  

# s='1'; beta=100; J=1.0; h0=0.0; Jz=0.5; h=0; Jm=1; Jp=np.linspace(0,1,10)

# print(Sz(beta,J,h0,Jz,h,Jm,Jp))