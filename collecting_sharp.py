import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import mu_n_sharp0, mu_n_sharp, rhoj_sharp, rhoj_sharp_t, jdos_mu_sharp

# fixed parameters for ZnTeO
m0 = 0.117  # in me
V = 2.8  # in eV
x0 = 0.01  # the alloy fraction
Eg = 2.25  # band gap in eV
Eo = 1.98  # Oxygen level in eV

# other parameters
n0 = 1e-6  # A^(-3) = x 10^24 cm^(-3)
n = [1, 3, 5, 7, 9]  # set of density levels
Ed = Eo - Eg  # eV

mu = np.zeros(1)
mu[0] = mu_n_sharp(n0, Ed, v=V, x=x0, m=m0)  # finding mu

# finding corresponding mus
'''
f_mu = open('Data/mu_ZeTeO'+str(x0)+'_n018.txt', 'w')  # open file for mu
for i in range(5):
    mu[i] = mu_n_sharp(n[i]*n0, Ed, v=V, x=x0, m=m0)  # finding mu
    f_mu.write(str(n[i]) + '*' + str(n0) + ' ' + str(mu[i]) + '\n')  # write in mu
print('mu done')
f_mu.close()
'''

# mu_max = mu_n_sharp(10*n0, Ed)  # finding mu_max

# useful energies
Ep0 = (Ed + np.sqrt(Ed**2 + 4*V**2*x0)) / 2  # E+(0)
Em0 = (Ed - np.sqrt(Ed**2 + 4*V**2*x0)) / 2  # E-(0)

# Ekf = mu_max + V**2*x0 / (Ed - mu_max)  # Ekf
# Epkf = (Ekf + Ed + np.sqrt((Ekf - Ed)**2 + 4*V**2*x0)) / 2  # Ep(kf)

# limits of the spectrum
Omega_min = Ep0 - Ed
Omega_max = 1.75  # eV VB -> Em

# forming the grids
Omega = np.linspace(0, Omega_max, 500)
Rhoj = np.zeros((1, len(Omega)))
Dj = np.zeros((1, len(Omega)))


# calculating rho_j and Dj
for j in range(1):
    # open files
    f_rho = open('Data/rhoj_ZnTeO'+str(x0)+'_n018.txt', 'w')  # open file for rhoj
    f_D = open('Data/Dj_ZnTeO'+str(x0)+'_n018.txt', 'w')  # open file for Dj
    for i in range(len(Omega)):
        Rhoj[j, i] = rhoj_sharp_t(Omega[i], Ed, mu[j], v=V, x=x0, m=m0)  # calculate rhoj
        Dj[j, i] = jdos_mu_sharp(Omega[i], mu[j], Ed, v=V, x=x0, m=m0)  # calculate Dj
        f_rho.write(str(Omega[i]) + ' ' + str(Rhoj[j, i]) + '\n')  # write in rhoj
        f_D.write(str(Omega[i]) + ' ' + str(Dj[j, i]) + '\n')  # write in Dj
    f_rho.close()
    f_D.close()
    print(j)
