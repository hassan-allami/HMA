import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import mu_n_broad, rhoj_t, dj_filling_1, dj_filling_approx

# fixed parameters
m0 = 0.09  # in me
V = 2.2  # in eV
x0 = 0.01  # the alloy fraction

# other parameters
n0 = 1e-6  # A^(-3) = x 10^24 cm^(-3)
n = [1, 3, 5, 7, 9]  # set of density levels
Ed = 0.38  # eV
Gamma = 10e-3  # 10 meV

mu = np.zeros(1)
mu[0] = mu_n_broad(n0, Ed, Gamma, v=V, x=x0, m=m0)  # finding mu
print(mu)
# finding corresponding mus
'''
f_mu = open('Data/mu_Ed02.txt', 'w')  # open file for mu
for i in range(5):
    mu[i] = mu_n_broad(n[i]*n0, Ed, Gamma)  # finding mu
    f_mu.write(str(n[i]) + '*' + str(n0) + ' ' + str(mu[i]) + '\n')  # write in mu
print('mu done')
f_mu.close()
'''

# mu_max = mu_n_broad(10*n0, Ed, Gamma)  # finding mu_max

# useful energies
Ep0 = (Ed + np.sqrt(Ed**2 + 4*V**2*x0)) / 2  # E+(0)
Em0 = (Ed - np.sqrt(Ed**2 + 4*V**2*x0)) / 2  # E-(0)

# Ekf = mu_max + V**2*x0 / (Ed - mu_max)  # Ekf
# Epkf = (Ekf + Ed + np.sqrt((Ekf - Ed)**2 + 4*V**2*x0)) / 2  # Ep(kf)

# limits of the spectrum
Omega_min = Ep0 - Ed
Omega_max = 1.75  # Epkf - Em0 + 0.5

# forming the grids
Omega = np.linspace(0, Omega_max, 500)
Rhoj = np.zeros((1, len(Omega)))
Dj = np.zeros((1, len(Omega)))


# calculating rho_j and Dj
for j in range(1):
    # open files
    f_rho = open('Data/rhoj_CdTeO_'+str(x0)+'_Gamma10_n018.txt', 'w')  # open file for rhoj
    f_D = open('Data/Dj_CdTeO_'+str(x0)+'_Gamma10_n018(new).txt', 'w')  # open file for Dj
    for i in range(len(Omega)):
        Rhoj[j, i] = rhoj_t(Omega[i], Ed, mu[j], Gamma,  v=V, x=x0, m=m0)  # calculate rhoj
        Dj[j, i] = dj_filling_approx(Omega[i], mu[j], Ed, Gamma,  v=V, x=x0, m=m0)  # calculate Dj
        f_rho.write(str(Omega[i]) + ' ' + str(Rhoj[j, i]) + '\n')  # write in rhoj
        f_D.write(str(Omega[i]) + ' ' + str(Dj[j, i]) + '\n')  # write in Dj
        print(i)
    f_rho.close()
    f_D.close()
    print(mu)
