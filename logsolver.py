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
    logWvec = -beta * (ext.V - mu) #Eq. right after Eq.(2.7)
    
    #Emat is already multiplied by dx, do not multiply it again later on
    separation = np.subtract.outer(ext.xg, ext.xg) #Prepare to create matrix
    logEmat = inter.log_integrated_boltzmann_over_cell(separation, ext.dx, beta) #Eq. (2.1)

    n = ext.nsize
    log_left = np.ones(n)
    log_right = np.ones(n)

    '''
    For hard-core interaction potentials, when solving upward or downward the triangular matrix,
    there is no need to "divide by the denominator" because it is of the form den = 1 - dx*Emat[x,x]*Wvec[x]
    and Emat[x,x] is always zero, so den=1 and the division is unnecessary. Because I would have to calculate
    the cell-integrated denominator, that is one less problem to tackle in the code    
    '''
    
    def lse(a):   # logaddexp que tolera un array vacío
        return -np.inf if a.size == 0 else np.logaddexp.reduce(a)
    
    #The following is the implementation to solve matrices from Eq.(2.11)
    for i in range(n):
        log_left[i] = np.logaddexp(0.0, lse(logEmat[i, :i] + logWvec[:i] + log_left[:i]))
    
    for i in range(n - 1, -1, -1):
        log_right[i] = np.logaddexp(0.0, lse(logEmat[i+1:, i] + logWvec[i+1:] + log_right[i+1:]))


    log_grand_partition_left = np.logaddexp(0.0, np.log(ext.dx) + lse(logWvec + log_left)) #I actually do not remember where I got this from. Now I cannot find it on the paper
    log_grand_partition_right = np.logaddexp(0.0, np.log(ext.dx) + lse(logWvec + log_right)) #I actually do not remember where I got this from. Now I cannot find it on the paper

    print(log_grand_partition_left, log_grand_partition_right) #They should be the same. At some point implement an error if they are not the same
    density = np.exp(logWvec + log_left + log_right - log_grand_partition_left) #Eq. (2.7)

    #np.savetxt(indata.out_file+"/dprof_T_"+str(ext.temp)+"_mu_"+str(ext.mu)+"_sw_r0_"+str(inter.r0)+"_epsilon_"+str(inter.eps)+".dat", np.column_stack((ext.xg, density)))
    return ext.xg, density, log_grand_partition_left


if __name__ == "__main__":
    
    #Set sw interaction potential
    inter = Interaction("square-well", sigma=1.0, r0=1.8, epsilon=1.0)
    
    #Set length and spacing of external potential
    length = 12.0
    dx = 0.02
    
    #Set range of parameters
    mu_max = -5.0
    mu_min = 5.0
    temp_min = 0.001
    temp_max = 100.0
    
    #Set number of calculations
    ncal = 10

    #Create all solutions
    for i in range(ncal):
        #Uniform sampling for mu
        mu = np.random.uniform(mu_min, mu_max)

        #Change to log sampling in temperature to cample across scales. Not sure which sampling is better.
        #temp = np.random.uniform(temp_min, temp_max)
        temp = 10**np.random.uniform(np.log10(0.001), np.log10(100))

        #Generate random external potential
        Vext = ExternalPotential(length=length, dx=dx)
        Vext.generate_Vext()
        #Vext.load_file("test.dat")
        
        #Get the theoretical density
        t01 = time.time()
        xg, density, log_xi_t = solve(inter, Vext, mu, 1.0/temp)
        t02 = time.time()
        nparticles = np.trapezoid(density, xg)
        #Print number of particles
        print("********  "+str(i)+"  **********")
        print(f"mu = {mu:.12g}")
        print(f"T = {temp:.12g}")
        print(f"log Xi = {log_xi_t:.12g}")
        print(f"Xi = {np.exp(log_xi_t):.12g}")
        print(f"<N> = {nparticles:.12g}")
        
        print("Execution time: ", t02-t01, " seconds")
        
        head = "mu = "+str(mu)+",   T = "+str(temp)+"\nnparticles = "+str(nparticles)+",  log_Xi_total = "+str(log_xi_t)+"\n\nx  beta*(\mu-Vext) Vext  density(theory)\n"
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
