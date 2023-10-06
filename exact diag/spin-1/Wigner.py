from hamiltonian import *

def state(h,s,N):
    d0=2
    d1=int(2*s +1)
    d = d0*d1*d0
    
    psi_r=Hamiltonian(h,s,N).reshape(d,d**(N-1))
    psi_c=np.conj(psi_r)
    rdm=ncon([psi_r,psi_c],((-1,1),(-2,1)))
    rdm=Qobj(ncon([psi_r,psi_c],((-1,1),(-2,1))),dims=[[2,d1,2],[2,d1,2]])
    return rdm


def wigner(t,f,h,s,N):
    u1=(1j*sigmaz()*f).expm() * (1j*sigmay()*t).expm()
    Pi_1=qeye(2)-np.sqrt(3)*sigmaz()
    delta_1=(1/2) * (u1*Pi_1*(u1.dag()))

    u2=(1j*jmat(s,'z')*f).expm() * (1j*jmat(s,'y')*t).expm()
    eta=np.sqrt(2*s*(2*s+1)*(2*s+2)/2)
    d=int(2*s+1)
    Pi_2=np.zeros((d,d))
    for i in range(d-1):
        Pi_2[i,i]=1-eta*np.sqrt(2/(d*(d-1)))
    Pi_2[d-1,d-1]=1+eta*np.sqrt(2*(d-1)/d)
    delta_2=(1/d) * (u2*Qobj(Pi_2)*(u2.dag()))
 
    delta = tensor(delta_1,delta_2,delta_1)
    return abs(np.real((state(h,s,N)*delta).tr())) - np.real((state(h,s,N)*delta).tr())

# lower bound for bipartite system
def EoF(h,s,N):
    d=int(2*s + 1)
    eof=[]
    for i in range(0,d) :
        for j in range(i+1,d) :
            S=np.zeros((2*d,2*d))
            S[i,(j+d)]=S[(j+d),i]=1
            S[j,(i+d)]=S[(i+d),j]=-1
            m=(state(h,s,N).sqrtm())*Qobj(S,dims=[[2,d],[2,d]])*(state(h,s,N).conj())*Qobj(S,dims=[[2,d],[2,d]])*(state(h,s,N).sqrtm())
            eig_vals=m.eigenenergies(sort='high',eigvals=4)
            c=max(0,np.sqrt(eig_vals[0])-np.sqrt(eig_vals[1])-np.sqrt(eig_vals[2])-np.sqrt(eig_vals[3]))
            eof+=[c**2]
    return np.real(np.sqrt(sum(eof)))


#lower bound tripartie system
def tau3(h,s,N):
    d=int(2*s + 1)
    eof=[]
    for i in range(int(2*d*(2*d - 1)/2 )):
        for j in range(1,1):
            L=np.zeros((2*d,2*d))
            L[i,(j+d)]=L[(j+d),i]=1
            L[j,(i+d)]=L[(i+d),j]=-1
            S=tensor(Qobj(L,dims=[[2,d],[2,d]]),sigmax())
            m=(state(h,s,N).sqrtm())*S*(state(h,s,N).conj())*S*(state(h,s,N).sqrtm())
            eig_vals=m.eigenenergies(sort='high',eigvals=4)
            c=max(0,np.sqrt(eig_vals[0])-np.sqrt(eig_vals[1])-np.sqrt(eig_vals[2])-np.sqrt(eig_vals[3]))
            eof+=[c**2]
    for i in range(6):
        for j in range(int(d*(d-1)/2)):
            L=np.zeros((4,4))
            L[i,(j+d)]=L[(j+d),i]=1
            L[j,(i+d)]=L[(i+d),j]=-1
            L2=np.zeros((3,3))
            L2[i,(j+d)]=L2[(j+d),i]=1
            L2[j,(i+d)]=L2[(i+d),j]=-1
            S=tensor(Qobj(L,dims=[[4,4],[4,4]]),Qobj(L2,dims=[[3,3],[3,3]]))
            m=(state(h,s,N).sqrtm())*S*(state(h,s,N).conj())*S*(state(h,s,N).sqrtm())
            eig_vals=m.eigenenergies(sort='high',eigvals=4)
            c=max(0,np.sqrt(eig_vals[0])-np.sqrt(eig_vals[1])-np.sqrt(eig_vals[2])-np.sqrt(eig_vals[3]))
            eof+=[c**2]
    return np.real(np.sqrt(sum(eof)))

print(tau3(2,1,4))