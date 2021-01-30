import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import jdos, jdos_sharp, gamma

# The constants
cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in eV^(-3/2)A^(-3)
m = 0.1  # in me
V = 2.8  # in eV
x = 0.01  # the alloy fraction

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# general variables
Ed = np.arange(-0.1, 0.11, 0.02)  # Ed steps
eta = 1e-4  # fix eta = 0.1 meV
d = 0.5  # step for stack plotting

# stack of Gamma = 0
E = np.linspace(0.52, 0.65, 500)
Dj = np.zeros((len(Ed), len(E)))
fig1, ax1 = plt.subplots(figsize=(7, 20))
for i in range(len(Ed)):
    for j in range(len(E)):
        Dj[i, j] = jdos_sharp(E[j], Ed[i])
    ax1.plot(E, 1e4*Dj[i, :] + i*d)
    print('Gamma = 0 ', i)

ax1.set_aspect(0.03)
plt.xlim(0.52, 0.65)
plt.ylim(0, 8.1)
ax1.set_yticks([])
ax1.set_xlabel('E [eV]', fontsize=16)
ax1.set_ylabel('$D_j(E)$ [a.u.]', fontsize=16)
ax1.tick_params(labelsize=16)

# stack of Gamma = 1 meV
Gamma = 2e-3
fig2, ax2 = plt.subplots(figsize=(7, 20))
for i in range(len(Ed)):
    for j in range(len(E)):
        Dj[i, j] = jdos(E[j], Ed[i], Gamma)
    ax2.plot(E, 1e4*Dj[i, :] + i*d)
    print('Gamma = fix ', i)

ax2.set_aspect(0.03)
plt.xlim(0.52, 0.65)
plt.ylim(0, 8.1)
ax2.set_yticks([])
ax2.set_xlabel('E [eV]', fontsize=16)
ax2.set_ylabel('$D_j(E)$ [a.u.]', fontsize=16)
ax2.tick_params(labelsize=16)

# stack of Ed-dependent Gamma
fig3, ax3 = plt.subplots(figsize=(7, 20))
for i in range(len(Ed)):
    for j in range(len(E)):
        if gamma(Ed[i], eta) < 1e-5:
            Dj[i, j] = jdos_sharp(E[j], Ed[i])
        else:
            Dj[i, j] = jdos(E[j], Ed[i], gamma(Ed[i], eta))
    ax3.plot(E, 1e4*Dj[i, :] + i*d)
    print('Gamma = changing ', i)

ax3.set_aspect(0.03)
plt.xlim(0.52, 0.65)
plt.ylim(0, 8.1)
ax3.set_yticks([])
ax3.set_xlabel('E [eV]', fontsize=16)
ax3.set_ylabel('$D_j(E)$ [a.u.]', fontsize=16)
ax3.tick_params(labelsize=16)

# plot Gamma for the last stack
ed = np.linspace(-0.1, 0.1, 1000)
Gamma = np.zeros(len(ed))
for i in range(len(ed)):
    Gamma[i] = gamma(ed[i], eta)
plt.figure()
plt.plot(1e3*ed, 1e3*Gamma)
plt.xlim(-100, 100)
plt.xlabel('$E_d$ [meV]', fontsize=18)
plt.ylabel('$\Gamma$ [meV]', fontsize=18)
plt.tick_params(labelsize=18)
plt.text(-75, 5, '$v = 5.6^3 \AA^3$', fontsize=20)
plt.text(-75, 4, r'$\beta = 0.2 $', fontsize=20)
plt.text(-75, 3, r'$\eta = 0.1$ meV', fontsize=20)

plt.show()
