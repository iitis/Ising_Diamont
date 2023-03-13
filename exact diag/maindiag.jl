using NCon, Printf, LinearAlgebra, LinearMaps, Arpack
include("doapplyham.jl");

model = "XX" 
N = 20
usePBC = true
numval = 1;

d = 2; # local dimension
sX = [0 1; 1 0]; sY = [0 -im; im 0]; sZ = [1 0; 0 -1]; sI = [1 0; 0 1];
if model == "ising"
    hloc = reshape(real(kron(sX,sX) + kron(sY,sY)),2,2,2,2);
    EnExact = -4/sin(pi/N); # for PBC
elseif model == "ising"
    hloc = reshape(-kron(sX,sX) + 0.5*kron(sI,sZ) + 0.5*kron(sZ,sI),2,2,2,2);
    EnExact = -2/sin(pi/(2*N)); # for PBC
end

doApplyHamClosed = LinearMap(psiIn -> doApplyHam(psiIn,hloc,N,usePBC), d^N;
  ismutating=false, issymmetric=true, ishermitian=true, isposdef=false)

##### Diagonalize Hamiltonian with eigs
diagtime = @elapsed Energy, psi = eigs(doApplyHamClosed; nev = numval,
  tol = 1e-12, which=:SR, maxiter = 300);

##### Check with exact energy
EnErr = Energy[1] - EnExact; # should equal to zero
@printf "NumSites: %d, Time: %d, Energy: %e, EnErr: %e \n" N diagtime Energy[1] EnErr
