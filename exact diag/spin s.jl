using LinearAlgebra, LinearMaps, Arpack, TensorOperations, QuantumInformation, Plots

include("doapplyham.jl");

N=2
usePBC = false
numval = 1;

d1=2; d2=3;

sx = [0 1/2; 1/2 0]; sy = [0 -im/2; im/2 0]; sz = [1/2 0; 0 -1/2]; si = [1 0; 0 1];
sX = [0 1/sqrt(2) 0; 1/sqrt(2) 0 1/sqrt(2); 0 1/sqrt(2) 0]; sY = [0 1/sqrt(2*im) 0;-1/sqrt(2*im) 0 1/sqrt(2*im); 0 -1/sqrt(2*im) 0]; 
sZ = [1 0 0; 0 0 0; 0 0 -1]; sI = [1 0 0; 0 1 0; 0 0 1];

function E(J::Int, h::Float64)
    hloc = reshape(J*(kron(sx,sX)+kron(sy,sY)+kron(sz,sZ)+kron(sX,sx)+kron(sY,sy)+kron(sZ,sz))-h*(kron(sz,sI)+kron(si,sZ)),d1,d2,d1,d2);
    doApplyHamClosed = LinearMap(psiIn -> doApplyHam(psiIn,hloc,N,usePBC), (d1*d2)^N;
    ismutating=false, issymmetric=true, ishermitian=true, isposdef=false)
  
    Energy, psi = eigs(doApplyHamClosed; nev = numval, tol = 1e-12, which=:SR, maxiter = 300);
  
    # state=reshape(psi,d1,d2,d1,d2);
    # r=ncon([state,state],[[-1,-2, 1,2],[-3,-4,1,2]],[false,true]);
    # rho=reshape(r,(d1*d2),(d1*d2));
    #cnc=max(0,abs(rho[2,3])-sqrt(rho[1,1]*rho[4,4]),abs(rho[1,4])-rho[2,2]);
    return  psi #negativity(rho,[2,3],2)
end

# J=1
# h=range(0,5,100)
# y=E.(γ,λ)

# plot(λ,y)

E(1,0.5)
