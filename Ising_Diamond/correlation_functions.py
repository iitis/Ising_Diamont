import numpy as np
from scipy import *
import matplotlib.pylab as plt


def l(J,si,sj,h):
    return J*(si+sj)+h
def g(J,si,sj,h0):
    return J*si*sj+(h0/2)*(si+sj)

## Boltzmann weights according to spin-s (Eq. 51 - Valverde et al 2008)
def w(s:str, si, sj, beta, J:float, h0:float, Jz:float, h:float, Jp:float):
    if s=='1/2':
        x=2*np.exp(-beta*(g(J,si,sj,h0) + Jz/4))*np.cosh(beta*np.sqrt(16*l(J,si,sj,h)**2)/4) + 2*np.exp(-beta*(g(J,si,sj,h0) - Jz/4))*np.cosh(beta*Jp/4)
    elif s=='1':
        x=np.exp(-beta*g(J,si,sj,h0))*(np.exp(beta*Jz)+2*np.exp(beta*Jz/2)*np.cosh(beta*np.sqrt(Jz**2 + 2*Jp**2)/2)+4*np.cosh(beta*Jp/2)*np.cosh(beta*l(J,si,sj,h))+
                                2*np.exp(-beta*Jz)*np.cosh(beta*2*l(J,si,sj,h)))
    return x
    
## magnetization <spin-s> (Eq. 30 - Carvalho et al 2019)
def mag(s:str,beta:float, J:float, h0:float, Jz:float, h:float, Jp:float):
    w0_tilda=w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)/abs(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))
    m=(1/2) * ((w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))/abs(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))) * (1/np.sqrt(1+4*w0_tilda**2))
    return m

def F(s:str, beta:float, J:float, h0:float, Jz:float, h:float, Jp:float):
    B=np.sqrt((w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))**2 + 4*w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)**2)
    lp=(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)+w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp)+B)/2
    free_energy=(-1/beta) * np.log(lp)
    return free_energy

# ## Pauli spin-spin correlation functions (Eq. 34 - Carvalho et al 2019)
def sigma_i_sigma_j(s:str,beta:float,r:int, J:float, h0:float, Jz:float, h:float, Jp:float):
    #s='1/2'
    B=np.sqrt((w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))**2 + 4*w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)**2)
    sisj=mag(s,beta,J,h0,Jz,h,Jp)**2 + (w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)/B)**2 * ((w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)+w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp)-B)/(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)+w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp)+B))**r
    return sisj

def w_z(s:str, si, sj, beta, J:float, h0:float, Jz:float, h:float, Jp:float):
    dw=(w(s,si,sj,beta,J,h0,Jz,h+0.0001,Jp)-w(s,si,sj,beta,J,h0,Jz,h,Jp))/0.0001
    return (1/(2*beta))*dw

def Sz(s:str,beta:float, J:float, h0:float, Jz:float, h:float,  Jp:float):
    B=np.sqrt((w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))**2 + 4*w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)**2)
    lp=(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)+w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp)+B)/2

    t1=(w_z(s,1/2,1/2,beta,J,h0,Jz,h,Jp)+w_z(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))/(2*lp)
    t2=((w_z(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w_z(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))*(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp)))/(2*B*lp)
    t3=(2*w_z(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)*w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp))/(B*lp)
    return t1+t2+t3

def SzSz(s:str,beta:float,r:int, J:float, h0:float, Jz:float, h:float,  Jp:float):
    B=np.sqrt((w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))**2 + 4*w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)**2)
    lp=(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)+w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp)+B)/2    
    lm=(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)+w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp)-B)/2
    u=lm/lp
    w0_tilda=(w_z(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)*(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))/B)-(
        w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)*(w_z(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))/B
    )
    szsz=Sz(s,beta,J,h0,Jz,h,Jp)**2 + (((w0_tilda)**2)/(lp*lm))*u**r
    return szsz

def Szsj(s:str,beta:float,r:int, J:float, h0:float, Jz:float, h:float,  Jp:float):
    B=np.sqrt((w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))**2 + 4*w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)**2)
    lp=(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)+w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp)+B)/2
    w0_tilda=(w_z(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)*(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))/B)-(
        w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)*(w_z(s,1/2,1/2,beta,J,h0,Jz,h,Jp)-w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp))/B
    )
    lm=(w(s,1/2,1/2,beta,J,h0,Jz,h,Jp)+w(s,-1/2,-1/2,beta,J,h0,Jz,h,Jp)-B)/2
    u=lm/lp
    szsj=Sz(s,beta,J,h0,Jz,h,Jp)*mag(s,beta,J,h0,Jz,h,Jp) - (w0_tilda*w(s,1/2,-1/2,beta,J,h0,Jz,h,Jp)*u**r)/(B*lm)
    return szsj

def SxSx(s:str,beta:float, J:float, h0:float, Jz:float, h:float,  Jp:float):
    dF=(F(s,beta,J,h0,Jz,h,Jp+0.0001)-F(s,beta,J,h0,Jz,h,Jp))/0.0001
    return dF
  