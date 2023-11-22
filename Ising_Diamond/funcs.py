import numpy as np
from correlation_functions import *
from qutip import *
from toqito.state_props import l1_norm_coherence, negativity, log_negativity
from toqito.state_props import concurrence as cnc
import statistics
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy

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
    #= args
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

def wigner_entropy(args):
    s, beta, J, h0, Jz, h, Jp = args
    t=np.linspace(0,np.pi/2,50); f=np.linspace(0,2*np.pi,50)
    data=list(wigner(ti, fi, s, beta, J, h0, Jz, h, Jp) for ti in t for fi in f)
    sigma=100
    wigner_smoothed = gaussian_filter1d(data, sigma, order=0, mode="mirror")
    #wigner_smoothed_normalized = wigner_smoothed / sum(wigner_smoothed)
    return entropy(wigner_smoothed,base=2)*(np.pi**2)/100

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
        for k in range(i + 1, n):
            generator = np.zeros((n, n),dtype=complex)
            generator[i, k] = -1j
            generator[k, i] = 1j
            generators.append(Qobj(generator))
    return generators

def lbc_3_qubits(args):
    """
    Compute the lower bound concurrence for a system of three qubits as described by Ming Li et al 2009 J. Phys. A: Math. Theor. 42 145303
    
    Parameters:
    s (string): spin half or one
    beta (float): inverse temperature
    J (float): coupling between Ising nodes
    h0, h (float): magnetic field
    Jz (float): coupling between Heisenberg nodes in the z-direction
    Jp (float): coupling between Heisenberg nodes in the x- and y-direction; Jp=Jx+Jy

    Returns:
    float: value of the lower bound concurrence
    """
    s,beta,J,h0,Jz,h,Jp=args
    c123=[]; c132=[]; c231=[]
    state=rho(s,beta,J,h0,Jz,h,Jp).ptrace([0,1,2])
    for gen in so(4):
        s123=tensor(gen,so(2)[0])
        s123.dims=[[2,2,2],[2,2,2]]
        eigen123=(state*(s123*state.conj()*s123)).eigenenergies(sort='high')
        eigen132=((state.permute([0,2,1]))*(s123*(state.permute([0,2,1])).conj()*s123)).eigenenergies(sort='high')
        eigen231=((state.permute([1,2,0]))*(s123*(state.permute([1,2,0])).conj()*s123)).eigenenergies(sort='high')
        c123.append(max(0,np.sqrt(eigen123[0])-np.sqrt(eigen123[1])-np.sqrt(eigen123[2])-np.sqrt(eigen123[3]))**2)
        c132.append(max(0,np.sqrt(eigen132[0])-np.sqrt(eigen132[1])-np.sqrt(eigen132[2])-np.sqrt(eigen132[3]))**2)
        c231.append(max(0,np.sqrt(eigen231[0])-np.sqrt(eigen231[1])-np.sqrt(eigen231[2])-np.sqrt(eigen231[3]))**2)
    lbc = (sum(c123) + sum(c132)+ sum(c231))/3
    return np.sqrt(lbc)

def lbc_4_qubits(args):
    """
    Compute the lower bound concurrence for a system of four qubits as described by Ming Li et al 2009 J. Phys. A: Math. Theor. 42 145303
    
    Parameters:
    s (string): spin half or one
    beta (float): inverse temperature
    J (float): coupling between Ising nodes
    h0, h (float): magnetic field
    Jz (float): coupling between Heisenberg nodes in the z-direction
    Jp (float): coupling between Heisenberg nodes in the x- and y-direction; Jp=Jx+Jy

    Returns:
    float: value of the lower bound concurrence
    """
    s,beta,J,h0,Jz,h,Jp=args
    c1_234=[]; c2_134=[]; c3_124=[]; c4_123=[]
    c12_34=[]; c13_24=[]; c14_23=[]
    state=rho(s,beta,J,h0,Jz,h,Jp)
    for gen in so(8):
        s1_p=tensor(so(2)[0],gen)
        s1_p.dims=[[2,2,2,2],[2,2,2,2]]
        eigen1_234=(state*(s1_p*state.conj()*s1_p)).eigenenergies(sort='high')
        eigen2_134=((state.permute([1,0,2,3]))*(s1_p*(state.permute([1,0,2,3])).conj()*s1_p)).eigenenergies(sort='high')    
        eigen3_124=((state.permute([2,0,1,3]))*(s1_p*(state.permute([2,0,1,3])).conj()*s1_p)).eigenenergies(sort='high')
        eigen4_123=((state.permute([3,0,1,2]))*(s1_p*(state.permute([3,0,1,2])).conj()*s1_p)).eigenenergies(sort='high')
        c1_234.append(max(0,np.sqrt(eigen1_234[0])-np.sqrt(eigen1_234[1])-np.sqrt(eigen1_234[2])-np.sqrt(eigen1_234[3]))**2)
        c2_134.append(max(0,np.sqrt(eigen2_134[0])-np.sqrt(eigen2_134[1])-np.sqrt(eigen2_134[2])-np.sqrt(eigen2_134[3]))**2)
        c3_124.append(max(0,np.sqrt(eigen3_124[0])-np.sqrt(eigen3_124[1])-np.sqrt(eigen3_124[2])-np.sqrt(eigen3_124[3]))**2)
        c4_123.append(max(0,np.sqrt(eigen4_123[0])-np.sqrt(eigen4_123[1])-np.sqrt(eigen4_123[2])-np.sqrt(eigen4_123[3]))**2)
    for gen1 in so(4):
        for gen2 in so(4):
            Sij_kl=tensor(gen1,gen2)
            Sij_kl.dims=[[2,2,2,2],[2,2,2,2]]
            eigen12_34=(state*(Sij_kl*state.conj()*Sij_kl)).eigenenergies(sort='high')
            eigen13_24=((state.permute([0,2,1,3]))*(Sij_kl*(state.permute([0,2,1,3])).conj()*Sij_kl)).eigenenergies(sort='high')
            eigen14_23=((state.permute([0,3,1,2]))*(Sij_kl*(state.permute([0,3,1,2])).conj()*Sij_kl)).eigenenergies(sort='high')
            c12_34.append(max(0,np.sqrt(eigen12_34[0])-np.sqrt(eigen12_34[1])-np.sqrt(eigen12_34[2])-np.sqrt(eigen12_34[3]))**2)
            c13_24.append(max(0,np.sqrt(eigen13_24[0])-np.sqrt(eigen13_24[1])-np.sqrt(eigen13_24[2])-np.sqrt(eigen13_24[3]))**2)
            c14_23.append(max(0,np.sqrt(eigen14_23[0])-np.sqrt(eigen14_23[1])-np.sqrt(eigen14_23[2])-np.sqrt(eigen14_23[3]))**2)
    lbc = (sum(c1_234) + sum(c2_134) + sum(c3_124) + sum(c4_123) + sum(c12_34) + sum(c13_24) + sum(c14_23) + sum(c12_34) + sum(c13_24) + sum(c14_23))/7
    return np.sqrt(lbc)

def lbc_3_mixed(args):
    """
    Compute the lower bound concurrence for a system of three qubits as described by Ming Li et al 2009 J. Phys. A: Math. Theor. 42 145303
    
    Parameters:
    s (string): spin half or one
    beta (float): inverse temperature
    J (float): coupling between Ising nodes
    h0, h (float): magnetic field
    Jz (float): coupling between Heisenberg nodes in the z-direction
    Jp (float): coupling between Heisenberg nodes in the x- and y-direction; Jp=Jx+Jy

    Returns:
    float: value of the lower bound concurrence
    """
    s,beta,J,h0,Jz,h,Jp=args
    c123=[]; c132=[]; c231=[]
    state=rho(s,beta,J,h0,Jz,h,Jp).ptrace([0,1,2])
    for gen1 in so(6):
        for gen2 in so(3):
            s123=tensor(gen1,gen2)
            s123.dims=[[2,3,3],[2,3,3]]
            eigen123=(state*(s123*state.conj()*s123)).eigenenergies(sort='high')
            eigen132=((state.permute([0,2,1]))*(s123*(state.permute([0,2,1])).conj()*s123)).eigenenergies(sort='high')
            c123.append(max(0,np.sqrt(eigen123[0])-np.sqrt(eigen123[1])-np.sqrt(eigen123[2])-np.sqrt(eigen123[3]))**2)
            c132.append(max(0,np.sqrt(eigen132[0])-np.sqrt(eigen132[1])-np.sqrt(eigen132[2])-np.sqrt(eigen132[3]))**2)
    for gen3 in so(9):
        s231=tensor(gen3,so(2)[0])
        s231.dims=[[3,3,2],[3,3,2]]
        eigen231=((state.permute([1,2,0]))*(s231*(state.permute([1,2,0])).conj()*s231)).eigenenergies(sort='high')
        c231.append(max(0,np.sqrt(eigen231[0])-np.sqrt(eigen231[1])-np.sqrt(eigen231[2])-np.sqrt(eigen231[3]))**2)
    lbc = (sum(c123) + sum(c132)+ sum(c231))/3
    return np.sqrt(lbc)

def lbc_4_mixed(args):
    """
    Compute the lower bound concurrence for a system of four qubits as described by Ming Li et al 2009 J. Phys. A: Math. Theor. 42 145303
    
    Parameters:
    s (string): spin half or one
    beta (float): inverse temperature
    J (float): coupling between Ising nodes
    h0, h (float): magnetic field
    Jz (float): coupling between Heisenberg nodes in the z-direction
    Jp (float): coupling between Heisenberg nodes in the x- and y-direction; Jp=Jx+Jy

    Returns:
    float: value of the lower bound concurrence
    """
    s,beta,J,h0,Jz,h,Jp=args
    c1_234=[]; c2_134=[]; c3_124=[]; c4_123=[]
    c12_34=[]; c13_24=[]; c14_23=[]
    state=rho(s,beta,J,h0,Jz,h,Jp)
    for gen1 in so(18):
        s1_p=tensor(so(2)[0],gen1)
        s1_p.dims=[[2,3,3,2],[2,3,3,2]]
        eigen1_234=(state*(s1_p*state.conj()*s1_p)).eigenenergies(sort='high')
        eigen4_123=((state.permute([3,0,1,2]))*(s1_p*(state.permute([3,0,1,2])).conj()*s1_p)).eigenenergies(sort='high')
        c1_234.append(max(0,np.sqrt(eigen1_234[0])-np.sqrt(eigen1_234[1])-np.sqrt(eigen1_234[2])-np.sqrt(eigen1_234[3]))**2)
        c4_123.append(max(0,np.sqrt(eigen4_123[0])-np.sqrt(eigen4_123[1])-np.sqrt(eigen4_123[2])-np.sqrt(eigen4_123[3]))**2)
    for gen2 in so(3):
        for gen3 in so(12):
            s2_p=tensor(gen2,gen3)
            s2_p.dims=[[2,3,3,2],[2,3,3,2]]
            eigen2_134=((state.permute([1,0,2,3]))*(s2_p*(state.permute([1,0,2,3])).conj()*s2_p)).eigenenergies(sort='high')    
            eigen3_124=((state.permute([2,0,1,3]))*(s2_p*(state.permute([2,0,1,3])).conj()*s2_p)).eigenenergies(sort='high')    
            c2_134.append(max(0,np.sqrt(eigen2_134[0])-np.sqrt(eigen2_134[1])-np.sqrt(eigen2_134[2])-np.sqrt(eigen2_134[3]))**2)
            c3_124.append(max(0,np.sqrt(eigen3_124[0])-np.sqrt(eigen3_124[1])-np.sqrt(eigen3_124[2])-np.sqrt(eigen3_124[3]))**2)
    for gen4 in so(6):
        for gen5 in so(6):
            Sij_kl=tensor(gen4,gen5)
            Sij_kl.dims=[[2,3,3,2],[2,3,3,2]]
            eigen12_34=(state*(Sij_kl*state.conj()*Sij_kl)).eigenenergies(sort='high')
            eigen13_24=((state.permute([0,2,1,3]))*(Sij_kl*(state.permute([0,2,1,3])).conj()*Sij_kl)).eigenenergies(sort='high')
            c12_34.append(max(0,np.sqrt(eigen12_34[0])-np.sqrt(eigen12_34[1])-np.sqrt(eigen12_34[2])-np.sqrt(eigen12_34[3]))**2)
            c13_24.append(max(0,np.sqrt(eigen13_24[0])-np.sqrt(eigen13_24[1])-np.sqrt(eigen13_24[2])-np.sqrt(eigen13_24[3]))**2)
    for gen6 in so(4):
        for gen7 in so(9):
            s14_23=tensor(gen6,gen7)
            s14_23.dims=[[2,3,3,2],[2,3,3,2]]
            eigen14_23=((state.permute([0,3,1,2]))*(s14_23*(state.permute([0,3,1,2])).conj()*s14_23)).eigenenergies(sort='high')
            c14_23.append(max(0,np.sqrt(eigen14_23[0])-np.sqrt(eigen14_23[1])-np.sqrt(eigen14_23[2])-np.sqrt(eigen14_23[3]))**2)
    lbc = (sum(c1_234) + sum(c2_134) + sum(c3_124) + sum(c4_123) + sum(c12_34) + sum(c13_24) + sum(c14_23) + sum(c12_34) + sum(c13_24) + sum(c14_23))/7
    return np.sqrt(lbc)

args=['1/2',300,1,1e-4,1,1e-4,1]
print(lbc_4_mixed(args))