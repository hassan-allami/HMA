import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import dj_filling_1, dj_filling_2, jdos, n_broad

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# fixed parameters and the grids
Ed = 0.2
G1 = 5e-3  # the first Gamma
G2 = 10e-3  # the second Gamma
mus = [-0.2, -0.1, 0, 0.1, 0.2]  # list of mu's
E = np.linspace(0.5, 0.7, 200)  # forming E grid
Dj = np.zeros((len(mus) + 1, len(E)))  # forming Dj's
mu = np.linspace(-0.3, 0.3, 100)  # forming mu grid for density plot
n = np.zeros((2, len(mu)))  # forming n(mu)


# form figure and axes
fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 8))

# plotting Djs for Gamma = 5 meV
for j in range(len(mus)):
    for i in range(len(E)):
        Dj[j, i] = dj_filling_1(E[i], mus[j], Ed, G1)
    ax1.plot(E, 1e4 * Dj[j, :], label='$\mu =$ '+str(mus[j])+' eV')
    print(j)
ax1.legend(frameon=False, fontsize=15)

# plotting Djs for Gamma = 10 meV
ax1.set_prop_cycle(None)
for j in range(len(mus)):
    for i in range(len(E)):
        Dj[j, i] = dj_filling_1(E[i], mus[j], Ed, G2)
    ax1.plot(E, 1e4 * Dj[j, :], '--')
    print(j)

# plotting the full Dj at T = 0 for reference
for i in range(len(E)):
    Dj[len(mus), i] = jdos(E[i], Ed, G1)
ax1.plot(E, 1e4 * Dj[len(mus), :], ':b')

for i in range(len(E)):
    Dj[len(mus), i] = jdos(E[i], Ed, G2)
ax1.plot(E, 1e4 * Dj[len(mus), :], ':b')

# configuring the Djs plots
ax1.set_xlim(0.5, 0.7)
ax1.set_xlabel('$E$ [eV]', fontsize=16)
ax1.set_ylabel(r'$D_j$ [$\times10^{20}{\rm eV}^{-1}{\rm cm}^{-3}$]', fontsize=16)
ax1.set_ylim(0, 3)
ax1.tick_params(labelsize=16)

# calculating density for the mu interval
for i in range(len(mu)):
    n[0, i] = n_broad(mu[i], Ed, G1)  # at Gamma = 5 meV
    n[1, i] = n_broad(mu[i], Ed, G2)  # at Gamma = 10 meV

# plotting density
ax2.plot(mu, 1e4*n[0, :], 'k', label='$\Gamma = 5$ meV')
ax2.plot(mu, 1e4*n[1, :], '--k', label='$\Gamma = 10$ meV')
ax2.set_xlim(-0.3, 0.3)
ax2.set_xlabel('$\mu$ [eV]', fontsize=16)
ax2.set_ylabel(r'$n$ [$\times10^{20}{\rm cm}^{-3}$]', fontsize=16)
ax2.set_ylim(0, 0.8)
ax2.tick_params(labelsize=16)
ax2.legend(frameon=False, fontsize=14)

plt.show()
