import numpy as np
from correlation_functions import *
from qutip import *
from toqito.state_props import l1_norm_coherence, negativity, log_negativity
from toqito.state_props import concurrence as cnc
import statistics
from scipy import signal
from scipy.stats import entropy
from scipy.stats import ortho_group

def rho(s,beta,J,h0,Jz,h,Jp):
    #s,beta,J,h0,Jz,h,Jp = args
    if s=='1/2':
        d=2
        sz=sigmaz()
        sx=sigmax()
        sy=sigmay()
    else:
        d=3
        sx=jmat(1,'x')
        sy=jmat(1,'y')
        sz=jmat(1,'z')
    id=tensor(qeye(2),qeye(d),qeye(d),qeye(2))
    single_site= mag(s,beta,J,h0,Jz,h,Jp)*(tensor(sigmaz(),qeye(d),qeye(d),qeye(2))+tensor(qeye(2),qeye(d),qeye(d),sigmaz())) + Sz(s,beta,J,h0,Jz,h,Jp) * (tensor(qeye(2),sz,qeye(d),qeye(2))+
                                                                                                                                               tensor(qeye(2),qeye(d),sz,qeye(2)))
    two_site = (sigma_i_sigma_j(s,beta,1,J,h0,Jz,h,Jp) * tensor(sigmaz(),qeye(d),qeye(d),sigmaz()) + SzSz(s,beta,1,J,h0,Jz,h,Jp) * tensor(qeye(2),sz,sz,qeye(2)) + 
                SxSx(s,beta,J,h0,Jz,h,Jp) * (tensor(qeye(2),sx,sx,qeye(2)) + tensor(qeye(2),sy,sy,qeye(2)) )+
                Szsj(s,beta,1,J,h0,Jz,h,Jp)*(tensor(sigmaz(),sz,qeye(d),qeye(2))+tensor(qeye(2),qeye(d),sz,sigmaz())+
                                              tensor(sigmaz(),qeye(d),sz,qeye(2))+tensor(qeye(2),sz,qeye(d),sigmaz())))
    
    three_site = mag(s,beta,J,h0,Jz,h,Jp)*SxSx(s,beta,J,h0,Jz,h,Jp) * (tensor (qeye(2),sx,sx,sigmaz()) + tensor (qeye(2),sy,sy,sigmaz())) + mag(s,beta,J,h0,Jz,h,Jp)*SzSz(s,beta,1,J,h0,Jz,h,Jp) * (tensor (qeye(2),sz,sz,sigmaz()) + tensor (sigmaz(),sz,sz,qeye(2)))
    + Sz(s,beta,J,h0,Jz,h,Jp)*sigma_i_sigma_j(s,beta,1,J,h0,Jz,h,Jp) * (tensor (sigmaz(),qeye(d),sz,sigmaz()) + tensor (sigmaz(), sz, qeye(d),sigmaz()))

    four_site = sigma_i_sigma_j(s,beta,1,J,h0,Jz,h,Jp) * SzSz(s,beta,1,J,h0,Jz,h,Jp) * tensor(sigmaz(),sz,sz,sigmaz()) + sigma_i_sigma_j(s,beta,1,J,h0,Jz,h,Jp) * SxSx(s,beta,J,h0,Jz,h,Jp) * (
        tensor(sigmaz(),sx,sx,sigmaz()) + tensor(sigmaz(),sy,sy,sigmaz())
    )
    state=  (1/(2*d*2*d)) * (id + single_site + two_site + three_site + four_site)
    return state

def rho_2(s:str,beta:float, J:float, h0:float, Jz:float, h:float, Jp:float):
    up = (1/4)*(1 + SzSz(s,beta,1,J,h0,Jz,h,Jp) + 2*(Sz(s,beta,J,h0,Jz,h,Jp)))
    uw = (1/4)*( 1 - SzSz(s,beta,1,J,h0,Jz,h,Jp))
    ux = (1/2) * SxSx(s,beta,J,h0,Jz,h,Jp)
    um = (1/4) *(1 + SzSz(s,beta,1,J,h0,Jz,h,Jp) - 2*(Sz(s,beta,J,h0,Jz,h,Jp)))
    #ro14 = (1/4) * (SxSx(s,beta,J,h0,Jz,h,Jp)- 0.3*SxSx(s,beta,J,h0,Jz,h,Jp))
    return 2*max(0,abs(ux)-np.sqrt(up*um),-uw)

def wigner(t, f, s, beta, J, h0, Jz, h, Jp):
    #t, f, s, beta, J, h0, Jz, h, Jp = args
    
    pi_1=qeye(2) - np.sqrt(3) * sigmaz(); U_1=((1j*sigmaz()*f).expm())*((1j*sigmay()*t).expm()); A_1=(U_1*pi_1*U_1.dag())/2
    pi_2=qeye(3)-2*Qobj([[1,0,0],[0,1,0],[0,0,-2]]); U_2=((1j*jmat(1,'z')*f).expm())*((1j*jmat(1,'y')*t).expm()); A_2=(U_2*pi_2*U_2.dag())/3

    if s=='1/2':
        wig=np.real((rho(s,beta,J,h0,Jz,h,Jp)*tensor(A_1,A_1,A_1,A_1)).tr())
    elif s=='1':
        wig=np.real((rho(s,beta,J,h0,Jz,h,Jp)*tensor(A_1,A_2,A_2,A_1)).tr())
    return wig

def neg(args):
    s, beta, J, h0, Jz, h, Jp = args
    t=np.linspace(0,np.pi/2,50); f=np.linspace(0,2*np.pi,50)
    return np.sum(list(map(lambda t: np.sum(list(map(lambda f : (np.abs(wigner(t, f, s, beta, J, h0, Jz, h, Jp))-wigner(t, f, s, beta, J, h0, Jz, h, Jp))*(1/np.pi)*np.sin(2*t),f))),t)))

def so(n):
    """
    Compute the generators of the SO(n) group.
    These are n*(n-1)/2 skew-symmetric matrices of size n x n.

    Parameters:
    n (int): Dimension of the SO(n) group

    Returns:
    list: List of n*(n-1)/2 skew-symmetric matrices
    """
    generators = []
    for i in range(n):
        for j in range(i + 1, n):
            generator = np.zeros((n, n))
            generator[i, j] = -1
            generator[j, i] = 1
            generators.append(Qobj(generator))
    return generators

def lbc_3_sites(args):
    s,beta,J,h0,Jz,h,Jp=args
    c123=[]; c132=[]; c231=[]
    state=rho(s,beta,J,h0,Jz,h,Jp).ptrace([0,1,2])
    for gen in so(4):
        s123=tensor(gen,so(2)[0])
        s123.dims=[[2,2,2],[2,2,2]]
        eigen123=(state*(s123*state.conj()*s123)).eigenenergies(sort='high')
        eigen132=((state.permute([0,2,1]))*(s123*(state.permute([0,2,1])).conj()*s123)).eigenenergies(sort='high')
        eigen231=((state.permute([1,2,0]))*(s123*(state.permute([1,2,0])).conj()*s123)).eigenenergies(sort='high')
        c123+=max(0,np.sqrt(eigen123[0])-np.sqrt(eigen123[1])-np.sqrt(eigen123[2])-np.sqrt(eigen123[3]))
        c132+=max(0,np.sqrt(eigen132[0])-np.sqrt(eigen132[1])-np.sqrt(eigen132[2])-np.sqrt(eigen132[3]))
        c231+=max(0,np.sqrt(eigen231[0])-np.sqrt(eigen231[1])-np.sqrt(eigen231[2])-np.sqrt(eigen231[3]))
    lbc = (c123)**2 + (c132)**2 + (c231)**2
    return lbc