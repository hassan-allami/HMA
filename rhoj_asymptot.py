import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import rhoj_sharp, rhoj, int_rhoj

# The constants
cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in eV^(-3/2)A^(-3)
m = 0.1  # in me
V = 2.8  # in eV
x = 0.01  # the alloy fraction

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# for Ed < 0
Ed = -0.2
mu = -0.3
Em0 = (Ed - np.sqrt(Ed ** 2 + 4 * V ** 2 * x)) / 2
Ep0 = (Ed + np.sqrt(Ed ** 2 + 4 * V ** 2 * x)) / 2
Omega = np.linspace(0.3, 0.8, 200)
rhoj_full = np.zeros(len(Omega))
rhoj_half = np.zeros(len(Omega))
rhoj_asym_full = cte ** 2 * m ** 3 * V * np.sqrt(x) * np.sqrt(1 - (Ep0 / Em0)) * (Omega + Em0) / (8 * np.pi ** 3)
rhoj_asym_half = cte ** 2 * m ** 3 * np.sqrt((mu - Em0) * (Ep0 - mu) / (Ed - mu)) * np.sqrt(1 - (Ep0 / Em0)) \
                 * ((Omega + mu - Ep0) ** (3/2)).real / (6 * np.pi ** 4)
for i in range(len(Omega)):
    rhoj_full[i] = rhoj_sharp(Omega[i], Ed, Ed)
    rhoj_half[i] = rhoj_sharp(Omega[i], Ed, mu)

plt.plot(Omega, 1e8 * rhoj_full, label='full $E_-$')
plt.plot(Omega, 1e8 * rhoj_half, label='partially filled $E_-$')
plt.plot(Omega, 1e8 * rhoj_asym_full, '--C0')
plt.plot(Omega, 1e8 * rhoj_asym_half, '--C1')
plt.axvline(x=Ep0 - Em0, ls='--', color='grey')
plt.text(0.6, 0.3, '$\sqrt{E_d^2 + 4V^2x}$', color='grey', fontsize=16)
plt.xlabel('$E \mp \hbar\omega_p$ [eV]', fontsize=16)
plt.ylabel(r'$\rho_j$ [$\times 10^{40}$ eV$^{-1}$cm$^{-6}$]', fontsize=16)
plt.tick_params(labelsize=16)
plt.xlim(0.3, 0.8)
plt.ylim(0, 0.5)
plt.legend(fontsize=16, frameon=False)

# for Ed > 0
Ed = 0.2
mu = 0
Gamma = 2e-2
Em0 = (Ed - np.sqrt(Ed ** 2 + 4 * V ** 2 * x)) / 2
Ep0 = (Ed + np.sqrt(Ed ** 2 + 4 * V ** 2 * x)) / 2
Omega = np.linspace(0, 1, 100)
rhoj_full = np.zeros(len(Omega))
rhoj_half = np.zeros(len(Omega))
rhoj_full_sharp = np.zeros(len(Omega))
rhoj_half_sharp = np.zeros(len(Omega))
rhoj_asym_full = cte ** 2 * m ** 3 * V * np.sqrt(x) * np.sqrt(1 - (Ep0 / Em0)) * (Omega + Em0) / (8 * np.pi ** 3)
rhoj_asym_half = cte ** 2 * m ** 3 * np.sqrt((mu - Em0) * (Ep0 - mu) / (Ed - mu)) * np.sqrt(1 - (Ep0 / Em0)) \
                 * ((Omega + mu - Ep0) ** (3/2)).real / (6 * np.pi ** 4)
for i in range(len(Omega)):
    rhoj_full[i] = rhoj(Omega[i], Ed, Ed+0.4, Gamma)
    rhoj_half[i] = rhoj(Omega[i], Ed, mu, Gamma)
    rhoj_full_sharp[i] = rhoj_sharp(Omega[i], Ed, Ed + 0.4)
    rhoj_half_sharp[i] = rhoj_sharp(Omega[i], Ed, mu)
    print(i)

plt.figure()
plt.plot(Omega, 1e8 * rhoj_full, label='full $E_-$')
plt.plot(Omega, 1e8 * rhoj_half, label='partially filled $E_-$')
plt.plot(Omega, 1e8 * rhoj_asym_full, '--C0')
plt.plot(Omega, 1e8 * rhoj_asym_half, '--C1')
plt.plot(Omega, 1e8 * rhoj_full_sharp, ':C0')
plt.plot(Omega, 1e8 * rhoj_half_sharp, ':C1')
plt.axvline(x=2*V*np.sqrt(x), ls='--', color='grey')
plt.text(0.57, 0.7, '$2V\sqrt{x}$', color='grey', fontsize=16)
plt.xlabel('$E \mp \hbar\omega_p$ [eV]', fontsize=16)
plt.ylabel(r'$\rho_j$ [$\times 10^{40}$ eV$^{-1}$cm$^{-6}$]', fontsize=16)
plt.tick_params(labelsize=16)
plt.xlim(0, 1)
plt.ylim(0, 1.4)
plt.legend(fontsize=16, frameon=False)

plt.show()
