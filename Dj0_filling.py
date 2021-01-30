import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import jdos_mu_sharp, n_sharp, nmax_sharp, jdos_sharp, kf_mu, mu_n_sharp

cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in eV^(-3/2)A^(-3)
kB = 8.617 * 1e-5  # Boltzmann constant in eV/Kelvin
m0 = 0.1  # in me
V = 2.8  # in eV
x0 = 0.01  # the alloy fraction

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)


''' for Ed < 0'''
# form figure and axes
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 8))

Ed = -0.2
n0 = 1e-6
mu_min = (Ed - np.sqrt(Ed**2 + 4*V**2*x0))/2 - 0.1
mu_max = Ed + 0.1
mu = np.linspace(mu_min, mu_max, 300)  # forming mu grid

nmax = nmax_sharp(Ed)  # max n
n = np.zeros((2, len(mu)))  # forming n(mu)
kf = np.zeros(len(mu))  # forming kf(mu)
for i in range(len(mu)):
    n[0, i] = n_sharp(mu[i], Ed, t=300)  # at T = 300 K
    n[1, i] = n_sharp(mu[i], Ed, t=10)  # at T = 10 K
    kf[i] = kf_mu(mu[i], Ed)  # kf(mu)

mu300 = mu_n_sharp(n0, Ed)
mu10 = mu_n_sharp(n0, Ed, t=10)

mus = [-0.4, -0.35, -0.3, -0.25, -0.2]  # list of mu's
E = np.linspace(0.5, 1.6, 300)  # forming E grid
Dj0 = np.zeros((len(mus) + 1, len(E)))  # forming Dj0s
Dj0_n300 = np.zeros(len(E))
Dj0_n10 = np.zeros(len(E))

# plotting Djs for T = 300 K
for j in range(len(mus)):
    for i in range(len(E)):
        Dj0[j, i] = jdos_mu_sharp(E[i], mus[j], Ed, t=300)
    ax1.plot(E, 1e4 * Dj0[j, :], label='$\mu =$ '+str(mus[j]) + ' eV')

# for n = 10 ^ 18
for i in range(len(E)):
    Dj0_n300[i] = jdos_mu_sharp(E[i], mu300, Ed, t=300)

# plot for n = 10 ^ 18
ax1.plot(E, 1e4 * Dj0_n300, ':k', label='$n = 10^{18}$ cm$^{-3}$')
ax1.legend(frameon=False, fontsize=15)

# plotting Djs for T = 10 K
ax1.set_prop_cycle(None)
for j in range(len(mus)):
    for i in range(len(E)):
        Dj0[j, i] = jdos_mu_sharp(E[i], mus[j], Ed, t=10)
    ax1.plot(E, 1e4 * Dj0[j, :], '--')

# for n = 10 ^ 18
for i in range(len(E)):
    Dj0_n10[i] = jdos_mu_sharp(E[i], mu10, Ed, t=10)
# plot for n = 10 ^ 18
ax1.plot(E, 1e4 * Dj0_n10, '--k')

# plotting the full Dj at T = 0 for reference
for i in range(len(E)):
    Dj0[len(mus), i] = jdos_sharp(E[i], Ed)
ax1.plot(E, 1e4 * Dj0[len(mus), :], ':b')

# configuring the Djs plots
ax1.axvline(x=np.sqrt(Ed**2 + 4*V**2*x0), ls='--', color='grey')
ax1.text(0.54, 0.08, '$\sqrt{Ed^2 + 4V^2x}$', fontsize=15, color='grey', rotation=90)
ax1.set_xlim(0.5, 1.6)
ax1.set_xlabel('$E$ [eV]', fontsize=16)
ax1.set_ylabel(r'$D_j$ [$\times10^{20}{\rm eV}^{-1}{\rm cm}^{-3}$]', fontsize=16)
ax1.set_ylim(0, 0.35)
ax1.tick_params(labelsize=16)

# plotting n
ax2.axvline(x=Ed, ls='--', color='grey')
ax2.axvline(x=(Ed - np.sqrt(Ed**2 + 4*V**2*x0))/2, ls='--', color='grey')
ax2.plot(mu, 1e4*n[0, :], 'k', label='$T = 300$ K')
ax2.plot(mu, 1e4*n[1, :], '--k', label='$T = 10$ K')
ax2.text(-0.32, 0.45, r'$n_{\rm max}$', fontsize=16, color='b')
ax2.text(-0.19, 0.1, '$E_d$', fontsize=16, color='grey')
ax2.text(-0.39, 0.1, '$E_-(0)$', fontsize=16, color='grey')
ax2.plot(mu, 1e4*nmax*np.ones(len(mu)), ':b')
ax2.set_aspect(0.4)
ax2.set_xlim(mu_min, mu_max)
ax2.set_xlabel('$\mu$ [eV]', fontsize=16)
ax2.set_ylabel(r'$n$ [$\times10^{20}{\rm cm}^{-3}$]', fontsize=16)
ax2.set_ylim(0, 0.5)
ax2.tick_params(labelsize=16)
ax2.legend(frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.5))

# plotting kf
ax22 = ax2.twinx()
ax22.plot(mu, kf, 'r')
ax22.set_ylim(0, 1)
ax22.spines['right'].set_color('r')
ax22.tick_params(labelsize=16, labelcolor='r')
ax22.set_ylabel('$k_F$ [$\AA ^ {-1}$]', fontsize=16, color='r')
ax22.axhline(y=0.5, ls=':', color='r')
ax22.text(-0.25, 0.1, '$k_F$', fontsize=16, color='r')


''' for Ed > 0'''
# form figure and axes
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 8))

Ed = 0.2
mu_min = (Ed - np.sqrt(Ed**2 + 4*V**2*x0))/2 - 0.1
mu_max = Ed + 0.1
mu = np.linspace(mu_min, mu_max, 300)  # forming mu grid

nmax = nmax_sharp(Ed)  # max n
n = np.zeros((2, len(mu)))  # forming n(mu)
kf = np.zeros(len(mu))  # forming kf(mu)
for i in range(len(mu)):
    n[0, i] = n_sharp(mu[i], Ed, t=300)  # at T = 300 K
    n[1, i] = n_sharp(mu[i], Ed, t=10)  # at T = 10 K
    kf[i] = kf_mu(mu[i], Ed)  # kf(mu)

Mu300 = mu_n_sharp(n0, Ed)
Mu10 = mu_n_sharp(n0, Ed, t=10)

mus = [-0.2, -0.1, 0, 0.1, 0.2]  # list of mu's
E = np.linspace(0.5, 0.9, 200)  # forming E grid
Dj0 = np.zeros((len(mus) + 1, len(E)))  # forming Dj0s
Dj0_n300 = np.zeros(len(E))
Dj0_n10 = np.zeros(len(E))

# plotting Djs for T = 300 K
for j in range(len(mus)):
    for i in range(len(E)):
        Dj0[j, i] = jdos_mu_sharp(E[i], mus[j], Ed, t=300)
    ax1.plot(E, 1e4 * Dj0[j, :], label = '$\mu =$ '+str(mus[j]) + ' eV')

# for n = 10 ^ 18
for i in range(len(E)):
    Dj0_n300[i] = jdos_mu_sharp(E[i], Mu300, Ed, t=300)

# plot for n = 10 ^ 18
ax1.plot(E, 1e4 * Dj0_n300, ':k', label='$n = 10^{18}$ cm$^{-3}$')
ax1.legend(frameon=False, fontsize=15)

# plotting Djs for T = 10 K
ax1.set_prop_cycle(None)
for j in range(len(mus)):
    for i in range(len(E)):
        Dj0[j, i] = jdos_mu_sharp(E[i], mus[j], Ed, t=10)
    ax1.plot(E, 1e4 * Dj0[j, :], '--')

# for n = 10 ^ 18
for i in range(len(E)):
    Dj0_n10[i] = jdos_mu_sharp(E[i], Mu10, Ed, t=10)
# plot for n = 10 ^ 18
ax1.plot(E, 1e4 * Dj0_n10, '--k')

# plotting the full Dj at T = 0 for reference
for i in range(len(E)):
    Dj0[len(mus), i] = jdos_sharp(E[i], Ed)
ax1.plot(E, 1e4 * Dj0[len(mus), :], ':b')

# configuring the Djs plots
ax1.axvline(x=2*V*np.sqrt(x0), ls='--', color='grey')
ax1.axvline(x=np.sqrt(Ed**2 + 4*V**2*x0), ls='--', color='grey')
ax1.text(0.535, 0.8, '$2V\sqrt{x}$', fontsize=15, color='grey', rotation=90)
ax1.text(0.6, 1.1, '$\sqrt{E_d^2 + 4V^2x}$', fontsize=15, color='grey', rotation=90)
ax1.set_xlim(0.5, 0.9)
ax1.set_xlabel('$E$ [eV]', fontsize=16)
ax1.set_ylabel(r'$D_j$ [$\times10^{20}{\rm eV}^{-1}{\rm cm}^{-3}$]', fontsize=16)
ax1.set_ylim(0, 2)
ax1.tick_params(labelsize=16)

# plotting n
ax2.axvline(x=Ed, ls='--', color='grey')
ax2.axvline(x=(Ed - np.sqrt(Ed**2 + 4*V**2*x0))/2, ls='--', color='grey')
ax2.plot(mu, 1e4*n[0, :], 'k', label='$T = 300$ K')
ax2.plot(mu, 1e4*n[1, :], '--k', label='$T = 10$ K')
ax2.text(0, 0.72, r'$n_{\rm max}$', fontsize=16, color='b')
ax2.text(0.21, 0.2, '$E_d$', fontsize=16, color='grey')
ax2.text(-0.19, 0.2, '$E_-(0)$', fontsize=16, color='grey')
ax2.plot(mu, 1e4*nmax*np.ones(len(mu)), ':b')
ax2.set_aspect(0.4)
ax2.set_xlim(mu_min, mu_max)
ax2.set_xlabel('$\mu$ [eV]', fontsize=16)
ax2.set_ylabel(r'$n$ [$\times10^{20}{\rm cm}^{-3}$]', fontsize=16)
ax2.set_ylim(-0.25, 1)  # this is being weird!
ax2.tick_params(labelsize=16)
ax2.legend(frameon=False, fontsize=14, bbox_to_anchor=(0.65, 0.5))

# plotting kf
kd = cte**(1/3) * np.sqrt(m0 * Ed)  # define kd
ax22 = ax2.twinx()
ax22.plot(mu, kf, 'r')
ax22.set_ylim(0, 1)
ax22.spines['right'].set_color('r')
ax22.tick_params(labelsize=16, labelcolor='r')
ax22.set_ylabel('$k_F$ [$\AA ^ {-1}$]', fontsize=16, color='r')
ax22.axhline(y=0.5, ls=':', color='r')
ax22.axhline(y=kd, ls=':', color='g')
ax22.text(0.12, 0.22, '$k_F$', fontsize=16, color='r')
ax22.text(0.12, 0.1, '$k_d$', fontsize=16, color='g')

print('mu at 300 =', mu300)
print('mu at 10 =', mu10)
print('mu at 300 =', Mu300)
print('mu at 10 =', Mu10)

plt.show()
