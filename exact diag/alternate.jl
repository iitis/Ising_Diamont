using ITensors
let
  N = 100
  sites = siteinds("S=1/2",N)

  os = OpSum()
  for j=1:N-1
    os += "Sz",j,"Sz",j+1
    os += 1/2,"S+",j,"S-",j+1
    os += 1/2,"S-",j,"S+",j+1
  end
  H = MPO(os,sites)

  psi0 = randomMPS(sites,10)

  nsweeps = 5
  outputlevel = 0
  maxdim = [10,20,100,100,200]
  cutoff = [1E-10]

  energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff, outputlevel)
  b=Int(N/2)
  orthogonalize!(psi, b)
  U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi,b)))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end
  return SvN/log(2)
end