import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import rhoj_sharp, rhoj_sharp_t, f, mu_n_sharp, mu_n_sharp0

cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in ev^(-3/2)A^(-3)
kB = 8.617 * 1e-5  # Boltzmann constant in ev/Kelvin
m0 = 0.1  # in me
V = 2.8  # in eV
x0 = 0.01  # the alloy fraction

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# fixed parameters and the grids
Ed = -0.2
n0 = 1e-6
mus = [-0.4, -0.35, -0.3, -0.25, -0.2]  # list of mu's
mu300 = mu_n_sharp(n0, Ed)
mu0 = mu_n_sharp0(n0, Ed)
print(mu300)
print(mu0)
Omega = np.linspace(0.3, 0.8, 100)  # forming Omega grid
Rhoj0 = np.zeros((len(mus), len(Omega)))  # forming rhoj0 grid for T = 0
Rhoj = np.zeros((len(mus), len(Omega)))  # forming rhoj0 grid for T = 300 K
Rhoj_asym = np.zeros((len(mus), len(Omega)))  # forming rhoj0 grid for the asymptotes
Rhoj_n300 = np.zeros(len(Omega))
Rhoj_n0 = np.zeros(len(Omega))


# the asymptotes
def rhoj0_asym(omega, mu):
    Ep0 = (Ed + np.sqrt(Ed ** 2 + 4 * V ** 2 * x0)) / 2  # E+(0)
    Em0 = (Ed - np.sqrt(Ed ** 2 + 4 * V ** 2 * x0)) / 2  # E-(0)
    coef1 = cte**2 * m0**3 / (8*np.pi**3)
    coef2 = V * np.sqrt(x0) * np.sqrt(1 - Ep0/Em0) * f(Ed - mu, t=300) * f(mu - Ep0, t=300)
    return coef1 * coef2 * (omega + Ed - Ep0)


# calculating and plotting rhoj0
for j in range(len(mus)):
    for i in range(len(Omega)):
        Rhoj0[j, i] = rhoj_sharp(Omega[i], ed=Ed, mu=mus[j])  # for T = 0
        Rhoj[j, i] = rhoj_sharp_t(Omega[i], ed=Ed, mu=mus[j])  # for T = 300
        Rhoj_asym[j, i] = rhoj0_asym(Omega[i], mu=mus[j])  # asymptotes
    plt.plot(Omega, 1e9*Rhoj0[j, :], '--C'+str(j))
    plt.plot(Omega, 1e9 * Rhoj_asym[j, :], ':C' + str(j))
    plt.plot(Omega, 1e9*Rhoj[j, :], '-C' + str(j), label='$\mu =$ '+str(mus[j])+' eV')

# for n = 10 ^ 18
for i in range(len(Omega)):
    Rhoj_n300[i] = rhoj_sharp_t(Omega[i], Ed, mu300, t=300)
    Rhoj_n0[i] = rhoj_sharp(Omega[i], Ed, mu0)

# plot for n = 10 ^ 18
plt.plot(Omega, 1e9 * Rhoj_n300, ':k', label='$n = 10^{18}$ cm$^{-3}$')
plt.plot(Omega, 1e9 * Rhoj_n0, '--k'    )

# configuring the plot
plt.xlabel('$E \mp \hbar\omega_p$ [eV]', fontsize=16)
plt.ylabel(r'$\rho_j^{(0)}$ [$\times 10^{41}$ eV$^{-1}$cm$^{-6}$]', fontsize=16)
plt.tick_params(labelsize=16)
plt.xlim(0.35, 0.8)
plt.ylim(0, 1)
plt.legend(fontsize=16, frameon=True, loc=1)

plt.show()
