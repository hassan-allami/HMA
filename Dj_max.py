import numpy as np
from matplotlib import pyplot as plt
from HMAfunctions import dj_max

# The constants
cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in eV^(-3/2)A^(-3)
m = 0.1  # in me
V = 2.8  # in eV
x = 0.01  # the alloy fraction

# plotting Dj_max and E_max as Ed changes
# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

eta = [1e-3, 1e-4, 1e-5]
Edmin, Edmax = -0.5, 0.5
Ed = np.linspace(Edmin, Edmax, 500)
Dj_max = np.zeros((3, len(Ed)))
E_max = np.zeros((3, len(Ed)))

for i in range(len(Ed)):
    E_max[0, i], Dj_max[0, i] = dj_max(Ed[i], eta[0])
    E_max[1, i], Dj_max[1, i] = dj_max(Ed[i], eta[1])
    E_max[2, i], Dj_max[2, i] = dj_max(Ed[i], eta[2])
    print(i)

fig, ax = plt.subplots()
ax.plot(1e3*Ed,  1e4*Dj_max[0, :], '-C0')
ax.plot(1e3*Ed,  1e4*Dj_max[1, :], '--C0')
ax.plot(1e3*Ed,  1e4*Dj_max[2, :], ':C0')
ax2 = ax.twinx()
ax2.plot(1e3*Ed, 1e3*(E_max[0, :] - 2*V*np.sqrt(x)), '-C1')
ax2.plot(1e3*Ed, 1e3*(E_max[1, :] - 2*V*np.sqrt(x)), '--C1')
ax2.plot(1e3*Ed, 1e3*(E_max[2, :] - 2*V*np.sqrt(x)), ':C1')
ax.set_xlabel('$E_d$ [meV]', fontsize=16)
ax.set_ylabel(r'$\max(D_j)$ [$\times 10^{20}$ eV$^{-1}$cm$^{-3}$]', fontsize=16, color='C0')
ax2.set_ylabel(r'$E_{\rm max} - 2V\sqrt{x}$ [meV]', fontsize=16, color='C1')
ax2.spines['right'].set_color('C1')
ax2.spines['left'].set_color('C0')
ax.tick_params(labelsize='large')
ax2.tick_params(labelsize='large')
plt.xlim(1e3*Edmin, 1e3*Edmax)

plt.show()
