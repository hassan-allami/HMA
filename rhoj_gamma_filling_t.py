import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import rhoj_t, rhoj_sharp_t, rhoj, mu_n_broad, mu_n_sharp, mu_n_broad0

V = 2.8  # in eV
x0 = 0.01  # the alloy fraction

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# fixed parameters and the grids
n0 = 1e-6  # A^(-3)
Ed = 0.2
Gamma = 10e-3
mus = [-0.2, -0.1, 0, 0.1, 0.2]  # list of mu's
# mus for n = 10^18
mu_gt = mu_n_broad(n0, Ed, Gamma)
mu_0t = mu_n_sharp(n0, Ed)
mu_g0 = mu_n_broad0(n0, Ed, Gamma)

Omega = np.linspace(0, 0.7, 100)  # forming Omega grid
Rhoj_gt = np.zeros((len(mus), len(Omega)))  # forming rhoj grid for finite Gamma finite T
Rhoj_0t = np.zeros((len(mus), len(Omega)))  # forming rhoj0 grid for zero Gamma finite T
Rhoj_g0 = np.zeros((len(mus), len(Omega)))  # forming rhoj grid for finite Gamma zero T
# forming rhoj grid for n = 10^18
Rhoj_gt_n = np.zeros(len(Omega))
Rhoj_0t_n = np.zeros(len(Omega))
Rhoj_g0_n = np.zeros(len(Omega))

# calculating and plotting rho_j
for j in range(len(mus)):
    for i in range(len(Omega)):
        Rhoj_gt[j, i] = rhoj_t(Omega[i], Ed, mus[j], Gamma)  # for finite Gamma finite T
        Rhoj_0t[j, i] = rhoj_sharp_t(Omega[i], Ed, mus[j])  # for Gamma = 0 finite T
        Rhoj_g0[j, i] = rhoj(Omega[i], Ed, mus[j], Gamma)  # for finite Gamma T = 0
    print(j)
    plt.plot(Omega, 1e9*Rhoj_gt[j, :], '-C'+str(j), label='$\mu =$ '+str(mus[j])+' eV')
    plt.plot(Omega, 1e9*Rhoj_0t[j, :], ':C' + str(j))
    plt.plot(Omega, 1e9*Rhoj_g0[j, :], '--C' + str(j))

# calculating and plotting rho_j for n = 10^18
for i in range(len(Omega)):
    Rhoj_gt_n[i] = rhoj_t(Omega[i], Ed, mu_gt, Gamma)
    Rhoj_0t_n[i] = rhoj_sharp_t(Omega[i], Ed, mu_0t)
    Rhoj_g0_n[i] = rhoj(Omega[i], Ed, mu_g0, Gamma)
print('n = 10^18 done')
plt.plot(Omega, 1e9*Rhoj_g0_n, '--k', label='$n = 10^{18}$ cm$^{-3}$')
plt.plot(Omega, 1e9*Rhoj_0t_n, ':k')
# configuring the plot
plt.xlabel('$E \mp \hbar\omega_p$ [eV]', fontsize=16)
plt.ylabel(r'$\rho_j$ [$\times 10^{41}$ eV$^{-1}$cm$^{-6}$]', fontsize=16)
plt.tick_params(labelsize=16)
plt.xlim(0, 0.7)
plt.ylim(0, 1)
plt.legend(fontsize=16, frameon=True, loc=1)

plt.show()
