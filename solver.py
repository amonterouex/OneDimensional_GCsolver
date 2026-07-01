from pathlib import Path
import numpy as np

from interaction import Interaction, ExternalPotential
from iofunctions import InData, read_parameters

'''
The implementation in this solver is based on Percus' 1982 paper:
"One-Dimensional Classical Fluid with Nearest-Neighbor Interaction in Arbitrary External Field"
'''
def solve(parameter_file: str = "parameters.dat") -> tuple[np.ndarray, np.ndarray, float]:
    indata = InData()
    read_parameters(parameter_file, indata)

    inter = Interaction(indata.type, sigma=1.0, r0=indata.r0, epsilon=indata.eps)
    ext = ExternalPotential(indata.pot_file, indata.mu, indata.temp)

    #W(x)
    Wvec = np.exp(-ext.beta * (ext.Vext - ext.mu)) #Eq. right after Eq.(2.7)
    
    #Emat is already multiplied by dx, do not multiply it again later on
    separation = np.subtract.outer(ext.xg, ext.xg) #Prepare to create matrix
    Emat = inter.integrated_boltzmann_over_cell(separation, ext.dx, ext.beta) #Eq. (2.1)

    n = ext.nsize
    left = np.ones(n)
    right = np.ones(n)

    '''
    For hard-core interaction potentials, when solving upward or downward the triangular matrix,
    there is no need to "divide by the denominator" because it is of the form den = 1 - dx*Emat[x,x]*Wvec[x]
    and Emat[x,x] is always zero, so den=1 and the division is unnecessary. Because I would have to calculate
    the cell-integrated denominator, that is one less problem to tackle in the code    
    '''
    #The following is the implementation to solve matrices from Eq.(2.11)
    for i in range(n):
        left[i] = 1.0 + np.dot(Emat[i, :i], Wvec[:i] * left[:i])
    
    for i in range(n - 1, -1, -1):
        right[i] = 1.0 + np.dot(Emat[i + 1 :, i], Wvec[i + 1 :] * right[i + 1 :])

    grand_partition = 1.0 + ext.dx * np.sum(Wvec * left) #I actually do not remember where I got this from. Now I cannot find it on the paper
    density = Wvec * left * right / grand_partition #Eq. (2.7)

    np.savetxt(indata.out_file+"/dprof_T_"+str(ext.temp)+"_mu_"+str(ext.mu)+"_sw_r0_"+str(inter.r0)+"_epsilon_"+str(inter.eps)+".dat", np.column_stack((ext.xg, density)))
    #output = Path(indata.out_file)
    #output.parent.mkdir(parents=True, exist_ok=True)
    #np.savetxt(output, np.column_stack((ext.xg, density)))
    return ext.xg, density, grand_partition


if __name__ == "__main__":
    import sys
    parameter_file = sys.argv[1] if len(sys.argv) > 1 else "parameters.dat"
    x, rho, xi = solve(parameter_file)
    print(f"Xi = {xi:.12g}")
    print(f"<N> = {np.trapezoid(rho, x):.12g}")
