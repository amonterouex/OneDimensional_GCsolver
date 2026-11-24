import time
import numpy as np
import matplotlib.pyplot as plt

from interaction import Interaction, ExternalPotential

'''
The implementation in this solver is based on Percus' 1982 paper:
"One-Dimensional Classical Fluid with Nearest-Neighbor Interaction in Arbitrary External Field"
'''
def solve(inter: Interaction, ext: ExternalPotential, mu:float, beta:float) -> tuple[np.ndarray, np.ndarray, float]:

    #W(x)
    Wvec = np.exp(-beta * (ext.V - mu)) #Eq. right after Eq.(2.7)
    
    #Emat is already multiplied by dx, do not multiply it again later on
    separation = np.subtract.outer(ext.xg, ext.xg) #Prepare to create matrix
    Emat = inter.integrated_boltzmann_over_cell(separation, ext.dx, beta) #Eq. (2.1)

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


    grand_partition_left = 1.0 + ext.dx * np.sum(Wvec * left)
    #grand_partition_right = 1.0 + ext.dx * np.sum(Wvec * right)

    density = Wvec * left * right / grand_partition_left #Eq. (2.7)

    #np.savetxt("dprof_T_"+str(ext.temp)+"_mu_"+str(ext.mu)+"_sw_r0_"+str(inter.r0)+"_epsilon_"+str(inter.eps)+".dat", np.column_stack((ext.xg, density)))

    return ext.xg, density, grand_partition_left


if __name__ == "__main__":
    
    #Set sw interaction potential
    inter = Interaction("square-well", sigma=1.0, r0=1.8, epsilon=1.0)
    
    #Set length and spacing of external potential
    length = 12.0
    dx = 0.01
    
    #Set range of parameters
    mu_max = -2.0
    mu_min = 2.0
    temp_min = 0.01
    temp_max = 10.0
    
    #Set number of calculations
    ncal = 1

    #Create all solutions
    for i in range(ncal):
        mu = np.random.uniform(mu_min, mu_max)
        temp = np.random.uniform(temp_min, temp_max)
        Vext = ExternalPotential(length=length, dx=dx)
        Vext.generate_Vext()
        
        #Get the theoretical density
        t01 = time.time()
        xg, density, xi_t = solve(inter, Vext, mu, 1.0/temp)
        t02 = time.time()
        nparticles = np.trapezoid(density, xg)
        #Print number of particles
        print("********  "+str(i)+"  **********")
        print(f"mu = {mu:.12g}")
        print(f"T = {temp:.12g}")
        print(f"Xi = {xi_t:.12g}")
        print(f"<N> = {nparticles:.12g}")
        
        print("Execution time: ", t02-t01, " seconds")
        
        head = "mu = "+str(mu)+",   T = "+str(temp)+"\nnparticles = "+str(nparticles)+",  Xi_total = "+str(xi_t)+"\n\nx  beta*(\mu-Vext) Vext  density(theory)\n"
        np.savetxt("d"+str(i).zfill(3)+".dat", np.column_stack((Vext.xg, -(Vext.V-mu)/temp, Vext.V, density)), header=head)
        #np.savetxt("d"+str(i).zfill(3)+".dat", np.column_stack((Vext.xg, Vext.V, density)), header=head)
        
        '''
        fig, axs = plt.subplots(2, 1) 
        axs[0].set_xlim((0,length))
        axs[0].plot(Vext.xg, Vext.V)
        
        axs[1].set_xlim((0,length))
        #axs[1].plot(xs, rho)
        axs[1].plot(Vext.xg, density)
        
        plt.show()
        '''
