import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import rhop_sharp, rhom_sharp, rhop, rhom

# The constants
cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in eV^(-3/2)A^(-3)
m = 0.1  # in me
V = 2.8  # in eV
x = 0.01  # the alloy fraction

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# plot for Ed = 0.2 eV
Ed = 0.2
Gamma = 2e-2
E = np.linspace(-0.4, 1, 500)
rho_p_sharp = np.zeros(len(E))
rho_m_sharp = np.zeros(len(E))
rho_p = np.zeros(len(E))
rho_m = np.zeros(len(E))
for i in range(len(E)):
    rho_p_sharp[i] = rhop_sharp(E[i], Ed)
    rho_m_sharp[i] = rhom_sharp(E[i], Ed)
    rho_p[i] = rhop(E[i], Ed, Gamma)
    rho_m[i] = rhom(E[i], Ed, Gamma)

plt.plot(E, 1e4*rho_p, '--')
plt.plot(E, 1e4*rho_m, '--')
plt.plot(E, 1e4*rho_p_sharp, 'C0', label=r'$\rho_+^{(0)}$')
plt.plot(E, 1e4*rho_m_sharp, 'C1', label=r'$\rho_-^{(0)}$')
plt.xlabel('$E$ [eV]', fontsize=16)
plt.ylabel(r'$\times 10^{20}$ eV$^{-1}$cm$^{-3}$', fontsize=16)
plt.tick_params(labelsize=16)
plt.xlim(-0.4, 1)
plt.ylim(0, 5)
plt.axvline(x=Ed, ls='--', color='k')
plt.axvline(x=(Ed + np.sqrt(Ed**2 + 4*V**2*x))/2, ls='--', color='k')
plt.axvline(x=(Ed - np.sqrt(Ed**2 + 4*V**2*x))/2, ls='--', color='k')
plt.text(0.41, 3, '$E_+(0)$', fontsize=16)
plt.text(-0.19, 3, '$E_-(0)$', fontsize=16)
plt.text(0.21, 4, '$E_d$', fontsize=16)
plt.legend(frameon=False, fontsize=16)

# plot for Ed = -0.2 eV
Ed = -0.2
Gamma = 2e-2
E = np.linspace(-0.6, 0.8, 500)
rho_p_sharp = np.zeros(len(E))
rho_m_sharp = np.zeros(len(E))
rho_p = np.zeros(len(E))
rho_m = np.zeros(len(E))
for i in range(len(E)):
    rho_p_sharp[i] = rhop_sharp(E[i], Ed)
    rho_m_sharp[i] = rhom_sharp(E[i], Ed)
    rho_p[i] = rhop(E[i], Ed, Gamma)
    rho_m[i] = rhom(E[i], Ed, Gamma)

plt.figure()
plt.plot(E, 1e4*rho_p, '--')
plt.plot(E, 1e4*rho_m, '--')
plt.plot(E, 1e4*rho_p_sharp, 'C0', label=r'$\rho_+^{(0)}$')
plt.plot(E, 1e4*rho_m_sharp, 'C1', label=r'$\rho_-^{(0)}$')
plt.xlabel('$E$ [eV]', fontsize=16)
plt.ylabel(r'$\times 10^{20}$ eV$^{-1}$cm$^{-3}$', fontsize=16)
plt.tick_params(labelsize=16)
plt.xlim(-0.6, 0.8)
plt.ylim(0, 5)
plt.axvline(x=Ed, ls='--', color='k')
plt.axvline(x=(Ed + np.sqrt(Ed**2 + 4*V**2*x))/2, ls='--', color='k')
plt.axvline(x=(Ed - np.sqrt(Ed**2 + 4*V**2*x))/2, ls='--', color='k')
plt.text(0.21, 3, '$E_+(0)$', fontsize=16)
plt.text(-0.39, 3, '$E_-(0)$', fontsize=16)
plt.text(-0.19, 4, '$E_d$', fontsize=16)
plt.legend(frameon=False, fontsize=16)


plt.show()
