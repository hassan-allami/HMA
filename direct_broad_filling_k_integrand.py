import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import epp, emm, app, amm, cp, cm, f

kB = 8.617 * 1e-5  # Boltzmann constant in eV/Kelvin
V = 2.8  # in eV
x0 = 0.01  # the alloy fraction

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)


# form figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# introduce fixed parameters
T = 300
Ed = 0.2
Gamma = 6e-3
mu1 = (3 * Ed - np.sqrt(Ed ** 2 + 4 * V ** 2 * x0)) / 2  # (Em0 + Ed)/2
mu0 = (Ed - np.sqrt(Ed ** 2 + 4 * V ** 2 * x0)) / 2  # Em0
E = (2 * V * np.sqrt(x0) + np.sqrt(Ed ** 2 + 4 * V ** 2 * x0)) / 2

# evaluate the functions for mu0
k = np.linspace(0, 1, 1000)  # the k grid
I1 = np.zeros(len(k), dtype='complex')
I2 = np.zeros(len(k), dtype='complex')
C1 = np.zeros(len(k))
C2 = np.zeros(len(k))
for i in range(len(k)):
    # main term in 1st form:
    I1[i] = app(k[i], Ed, Gamma) * amm(k[i], Ed, Gamma) * f(epp(k[i], Ed, Gamma) - E - mu0, T) *\
            f(mu0 - epp(k[i], Ed, Gamma), T) / (E - epp(k[i], Ed, Gamma) + emm(k[i], Ed, Gamma))
    # main term in 2nd form:
    I2[i] = app(k[i], Ed, Gamma) * amm(k[i], Ed, Gamma) * f(emm(k[i], Ed, Gamma) - mu0, T) *\
            f(mu0 - E - emm(k[i], Ed, Gamma), T) / (E - epp(k[i], Ed, Gamma) + emm(k[i], Ed, Gamma))
    # correction of the 1st form
    C1[i] = cp(k[i], Ed, Gamma, mu0, T, E).real/(2*np.pi*(1 - np.exp(- E/(kB*T))))
    # correction of the 2nd form
    C2[i] = cm(k[i], Ed, Gamma, mu0, T, E).real/(2*np.pi*(1 - np.exp(- E/(kB*T))))


# plotting the functions for mu0
ax1.plot(k, I1.imag + C1, 'k', label='full first form')
ax1.plot(k, I2.imag + C2, '--r', label='full second form')
ax1.plot(k, I1.imag, '--C0',
         label=r'${\rm Im}\left[\frac{a_+^+a_-^-}{E - E_+^+ + E_-^-}f(E_+^+ - E - \mu)f(\mu - E_+^+)\right]$')
ax1.plot(k, C1, ':C0',
         label=r'$\frac{{\rm Re}[C_+]}{2\pi (1-e^{-E/T})}$')
ax1.plot(k, I2.imag, '--C1',
         label=r'${\rm Im}	\left[\frac{a_+^+a_-^-}{E - E_+^+ + E_-^-}f(E_-^- - \mu)f(\mu - E - E_-^-)\right]$')
ax1.plot(k, C2, ':C1',
         label=r'$\frac{{\rm Re}[C_-]}{2\pi (1-e^{-E/T})}$')
ax1.axhline(y=0, ls='--', color='grey')
ax1.set_xlim(0, 1)
# ax1.set_ylim(-0.2, 2)
ax1.set_xlabel(r'$\hbar k / \sqrt{2m}$ [eV]$^{1/2}$', fontsize=16)
ax1.set_ylabel('1/eV', fontsize=16)
ax1.tick_params(labelsize=16)
ax1.legend(frameon=False, fontsize=10)


# evaluate the functions for mu1
k = np.linspace(0, 1, 1000)  # the k grid
I1 = np.zeros(len(k), dtype='complex')
I2 = np.zeros(len(k), dtype='complex')
C1 = np.zeros(len(k))
C2 = np.zeros(len(k))
for i in range(len(k)):
    # main term in 1st form:
    I1[i] = app(k[i], Ed, Gamma) * amm(k[i], Ed, Gamma) * f(epp(k[i], Ed, Gamma) - E - mu1, T) *\
            f(mu1 - epp(k[i], Ed, Gamma), T) / (E - epp(k[i], Ed, Gamma) + emm(k[i], Ed, Gamma))
    # main term in 2nd form:
    I2[i] = app(k[i], Ed, Gamma) * amm(k[i], Ed, Gamma) * f(emm(k[i], Ed, Gamma) - mu1, T) *\
            f(mu1 - E - emm(k[i], Ed, Gamma), T) / (E - epp(k[i], Ed, Gamma) + emm(k[i], Ed, Gamma))
    # correction of the 1st form
    C1[i] = cp(k[i], Ed, Gamma, mu1, T, E).real/(2*np.pi*(1 - np.exp(- E/(kB*T))))
    # correction of the 2nd form
    C2[i] = cm(k[i], Ed, Gamma, mu1, T, E).real/(2*np.pi*(1 - np.exp(- E/(kB*T))))


# plotting the functions for mu0
ax2.plot(k, I1.imag + C1, 'k', label='full first form')
ax2.plot(k, I2.imag + C2, '--r', label='full second form')
ax2.plot(k, I1.imag, '--C0',
         label=r'${\rm Im}\left[\frac{a_+^+a_-^-}{E - E_+^+ + E_-^-}f(E_+^+ - E - \mu)f(\mu - E_+^+)\right]$')
ax2.plot(k, C1, ':C0',
         label=r'$\frac{{\rm Re}[C_+]}{2\pi (1-e^{-E/T})}$')
ax2.plot(k, I2.imag, '--C1',
         label=r'${\rm Im}	\left[\frac{a_+^+a_-^-}{E - E_+^+ + E_-^-}f(E_-^- - \mu)f(\mu - E - E_-^-)\right]$')
ax2.plot(k, C2, ':C1',
         label=r'$\frac{{\rm Re}[C_-]}{2\pi (1-e^{-E/T})}$')
ax2.axhline(y=0, ls='--', color='grey')
ax2.set_xlim(0, 1)
# ax2.set_ylim(-1, 4)
ax2.set_xlabel(r'$\hbar k / \sqrt{2m}$ [eV]$^{1/2}$', fontsize=16)
ax2.set_ylabel('1/eV', fontsize=16)
ax2.tick_params(labelsize=16)
# ax2.legend(frameon=False)

plt.show()
