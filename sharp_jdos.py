import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import jdos_sharp

# The constants
cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in eV^(-3/2)A^(-3)
m = 0.1  # in me
V = 2.8  # in eV
x = 0.01  # the alloy fraction


# plot jDOS for sharp momentum conserving transitions
# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

E = np.linspace(0.5, 1, 500)
JDOSp = np.zeros(len(E))
JDOSm = np.zeros(len(E))
for i in range(len(E)):
    JDOSp[i] = jdos_sharp(E[i], ed=0.2)
    JDOSm[i] = jdos_sharp(E[i], ed=-0.2)

# plot for Ed = 0.2 eV
plt.plot(E, 1e4 * JDOSp)
plt.xlabel('$E$ [eV]', fontsize=16)
plt.ylabel(r'$D_j$ [$\times 10 ^{20}$ eV$^{-1}$cm$^{-3}$]', fontsize=16)
plt.axvline(x=2 * V * np.sqrt(x), ls='--', color='k')
plt.text(0.54, 2.5, '$2V\sqrt{x}$', rotation=90, fontsize=14, color='k')
plt.axvline(x=np.sqrt(0.2 ** 2 + 4 * V ** 2 * x), ls='--', color='grey')
plt.text(0.6, 3.5, '$\sqrt{E_d^2 + 4V^2x}$', fontsize=14, color='grey')
plt.xlim(0.5, 1)
plt.ylim(0, 5)
plt.tick_params(labelsize='large')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# plot for Ed = -0.2 eV
plt.figure()
plt.plot(E, 1e5 * JDOSm)
plt.xlabel('$E$ [eV]', fontsize=16)
plt.ylabel(r'$D_j$ [$\times 10 ^{19}$ eV$^{-1}$cm$^{-3}$]', fontsize=16)
plt.axvline(x=2 * V * np.sqrt(x), ls='--', color='grey')
plt.text(0.54, 2.5, '$2V\sqrt{x}$', rotation=90, fontsize=14, color='grey')
plt.axvline(x=np.sqrt(0.2 ** 2 + 4 * V ** 2 * x), ls='--', color='k')
plt.text(0.6, 3.5, '$\sqrt{E_d^2 + 4V^2x}$', fontsize=14, color='k')
plt.xlim(0.5, 1)
plt.ylim(0, 5)
plt.tick_params(labelsize='large')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.show()
