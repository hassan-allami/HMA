import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import gamma, jdos

# The constants
cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in eV^(-3/2)A^(-3)
m = 0.1  # in me
V = 2.8  # in eV
x = 0.01  # the alloy fraction

# plot with moving Ed
# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)


E = np.linspace(0.54, 0.58, 500)
Dj = np.zeros((8, len(E)))
Ed = [-0.02, -0.01, 0, 0.0001, 0.001, 0.01, 0.05, 0.1]

for i in range(len(E)):
    Dj[0, i] = jdos(E[i], ed=Ed[0], gamma=gamma(Ed[0]))
    Dj[1, i] = jdos(E[i], ed=Ed[1], gamma=gamma(Ed[1]))
    Dj[2, i] = jdos(E[i], ed=Ed[2], gamma=gamma(Ed[2]))
    Dj[3, i] = jdos(E[i], ed=Ed[3], gamma=gamma(Ed[3]))
    Dj[4, i] = jdos(E[i], ed=Ed[4], gamma=gamma(Ed[4]))
    Dj[5, i] = jdos(E[i], ed=Ed[5], gamma=gamma(Ed[5]))
    Dj[6, i] = jdos(E[i], ed=Ed[6], gamma=gamma(Ed[6]))
    Dj[7, i] = jdos(E[i], ed=Ed[7], gamma=gamma(Ed[7]))
    print(i)

plt.figure()
plt.plot(E, 1e4 * Dj[0, :], label='$E_d = -20$ meV')
plt.plot(E, 1e4 * Dj[1, :], label='$E_d = -10$ meV')
plt.plot(E, 1e4 * Dj[2, :], label='$E_d = 0$')
plt.plot(E, 1e4 * Dj[3, :], label='$E_d = 0.1$ meV')
plt.plot(E, 1e4 * Dj[4, :], label='$E_d = 1$ meV')
plt.plot(E, 1e4 * Dj[5, :], label='$E_d = 10$ meV')
plt.plot(E, 1e4 * Dj[6, :], label='$E_d = 50$ meV')
plt.plot(E, 1e4 * Dj[7, :], label='$E_d = 100$ meV')
plt.axvline(x=2 * V * np.sqrt(x), ls='--', color='r')
plt.text(0.565, 2, '$2V\sqrt{x}$', fontsize=14, color='k', rotation=-90)
plt.xlabel('$E$ [eV]', fontsize=16)
plt.xlim(0.54, 0.58)
plt.ylabel(r'$D_j$ [$\times 10^{20}$ eV$^{-1}$cm$^{-3}$]', fontsize=16)
plt.ylim(0, 3)
plt.tick_params(labelsize='large')
plt.legend(fontsize=16, frameon=False)

# plot Gamma
Ed = np.linspace(-0.001, 0.001, 500)
Gamma = np.zeros(len(Ed))
for i in range(len(Ed)):
    Gamma[i] = gamma(Ed[i])

plt.figure()
plt.plot(1e3 * Ed, 1e3 * Gamma, 'k')
plt.xlabel('$E_d$ [meV]', fontsize=16)
plt.xlim(-1, 1)
plt.ylabel('$\Gamma$ [meV]', fontsize=16)
# plt.ylim(0, 17)
plt.tick_params(labelsize='large')


plt.show()
