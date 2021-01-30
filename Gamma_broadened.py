import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import gamma, jdos, jdos_sharp

# The constants
cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in eV^(-3/2)A^(-3)
m = 0.1  # in me
V = 2.8  # in eV
x = 0.01  # the alloy fraction

# plot with moving Ed
# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# plot for an Ed < 0 case
E = np.linspace(0.4, 1, 500)
Dj1 = np.zeros(len(E))
Dj2 = np.zeros(len(E))
Dj3 = np.zeros(len(E))
Dj_sharp = np.zeros(len(E))
for i in range(len(E)):
    Dj1[i] = jdos(E[i], ed=-0.2, gamma=0.01)
    Dj2[i] = jdos(E[i], ed=-0.2, gamma=0.02)
    Dj3[i] = jdos(E[i], ed=-0.2, gamma=0.03)
    Dj_sharp[i] = jdos_sharp(E[i], ed=-0.2)
    print(i)

plt.plot(E, 1e5 * Dj_sharp, 'k', label='$\Gamma = 0$')
plt.plot(E, 1e5 * Dj1, label='$\Gamma = 10$ meV')
plt.plot(E, 1e5 * Dj2, label='$\Gamma = 20$ meV')
plt.plot(E, 1e5 * Dj3, label='$\Gamma = 30$ meV')
plt.axvline(x=np.sqrt(0.2 ** 2 + 4 * V ** 2 * x), ls='--', color='r')
plt.text(0.6, 3.2, '$\sqrt{E_d^2 + 4V^2x}$', fontsize=14, color='k')
plt.xlabel('$E$ [eV]', fontsize=16)
plt.xlim(0.4, 1)
plt.ylabel(r'$D_j$ [$\times 10^{19}$ eV$^{-1}$cm$^{-3}$]', fontsize=16)
plt.ylim(0, 4)
plt.tick_params(labelsize='large')
plt.legend(fontsize=16, frameon=False)


# plot for an Ed > 0 case
E = np.linspace(0.5, 0.7, 500)
Dj1 = np.zeros(len(E))
Dj2 = np.zeros(len(E))
Dj3 = np.zeros(len(E))
Dj_sharp = np.zeros(len(E))
for i in range(len(E)):
    Dj1[i] = jdos(E[i], ed=0.2, gamma=0.01)
    Dj2[i] = jdos(E[i], ed=0.2, gamma=0.02)
    Dj3[i] = jdos(E[i], ed=0.2, gamma=0.03)
    Dj_sharp[i] = jdos_sharp(E[i], ed=0.2)
    print(i)

plt.figure()
plt.plot(E, 1e4 * Dj_sharp, 'k', label='$\Gamma = 0$')
plt.plot(E, 1e4 * Dj1, label='$\Gamma = 10$ meV')
plt.plot(E, 1e4 * Dj2, label='$\Gamma = 20$ meV')
plt.plot(E, 1e4 * Dj3, label='$\Gamma = 30$ meV')
plt.axvline(x=2*V*np.sqrt(x), ls='--', color='r')
plt.text(0.55, 2, '$2V\sqrt{x}$', fontsize=14, color='k', rotation=90)
plt.xlabel('$E$ [eV]', fontsize=16)
plt.xlim(0.5, 0.7)
plt.ylabel(r'$D_j$ [$\times 10^{20}$ eV$^{-1}$cm$^{-3}$]', fontsize=16)
plt.ylim(0, 3)
plt.tick_params(labelsize='large')
plt.legend(fontsize=16, frameon=False)

plt.show()
