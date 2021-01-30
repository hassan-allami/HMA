import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

kB = 8.617 * 1e-5  # Boltzmann constant in ev/Kelvin
wp = 26e-3  # hbar wp in eV (w_LO is used)
nB = 1/(np.exp(wp/(kB*300)) - 1)
print(nB)

x0 = 0.01  # the alloy fraction
v0 = 6. ** 3  # unit cell volume for ZeTe
C = [10, 20]  # the coefficient

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.subplots_adjust(wspace=0.3, left=0.1, right=0.96)
fig.suptitle('ZnTe$_{0.99}$O$_{0.01}$, $n = 10^{18}$ cm$^{-3}$, $E_d = -0.27$ eV, $V = 2.8$ eV, '
             '$\hbar\omega_p = 26$ meV', fontsize=18)

# open files
f_rho = open('Data/rhoj_ZnTeO' + str(x0) + '_n018.txt')  # open file for rhoj
f_D = open('Data/Dj_ZnTeO' + str(x0) + '_n018.txt')  # open file for Dj
# read data for rhoj
rho_data = f_rho.read()
rho_data = rho_data.split()
rho_data = np.array(list(map(float, rho_data)))
rho_data = np.reshape(rho_data, (int(len(rho_data) / 2), 2))
# making the shifts
E = rho_data[:, 0]  # the common energy axis
f_ab = interp1d(E - wp, rho_data[:, 1], kind='cubic', bounds_error=False)
f_em = interp1d(E + wp, rho_data[:, 1], kind='cubic', bounds_error=False)
rho_ab = f_ab(E)  # absorption
rho_em = f_em(E)  # emission

# read data for Dj
Dj_data = f_D.read()
Dj_data = Dj_data.split()
Dj_data = np.array(list(map(float, Dj_data)))
Dj_data = np.reshape(Dj_data, (int(len(Dj_data) / 2), 2))
# close the files
f_rho.close()
f_D.close()

# plot
ax1.plot(E, 1e6 * C[0] * v0 * rho_ab * nB, '-.C1', label=r'$\rho_j$ absorption')  # plot rhoj absorption
ax1.plot(E, 1e6 * C[0] * v0 * rho_em * (nB + 1), ':C1', label=r'$\rho_j$ emission')  # plot rhoj emission
ax1.plot(E, 1e6 * Dj_data[:, 1], '--C3', label='$D_j$')  # plot Dj
ax1.plot(E, 1e6 * (C[0] * v0 * ((nB+1)*rho_em + nB*rho_ab) + Dj_data[:, 1]), '-C0',
         label='$C =$ ' + str(C[0]))  # plot the sum
ax1.plot(E, 1e6 * (C[1] * v0 * ((nB+1)*rho_em + nB*rho_ab) + Dj_data[:, 1]), '-C4',
         label='$C =$ ' + str(C[1]))  # plot the sum
ax2.plot(E, 1e6 * C[0] * v0 * rho_ab * nB, '-.C1', label=r'$\rho_j$ absorption')  # plot rhoj
ax2.plot(E, 1e6 * C[0] * v0 * rho_em * (nB + 1), ':C1', label=r'$\rho_j$ emission')  # plot rhoj
ax2.plot(E, 1e6 * Dj_data[:, 1], '--C3', label='$D_j$')  # plot Dj
ax2.plot(E, 1e6 * (C[0] * v0 * ((nB+1)*rho_em + nB*rho_ab) + Dj_data[:, 1]), '-C0',
         label='$C =$ ' + str(C[0]))  # plot the sum
ax2.plot(E, 1e6 * (C[1] * v0 * ((nB+1)*rho_em + nB*rho_ab) + Dj_data[:, 1]), '-C4',
         label='$C =$ ' + str(C[1]))  # plot the sum
# add zoomed in inset
ax11 = fig.add_axes([0.79, 0.55, 0.155, 0.3])
ax11.plot(E, 1e6 * C[0] * v0 * rho_ab * nB, '-.C1', label=r'$\rho_j$ absorption')  # plot rhoj absorption
ax11.plot(E, 1e6 * C[0] * v0 * rho_em * (nB + 1), ':C1', label=r'$\rho_j$ emission')  # plot rhoj emission
ax11.plot(E, 1e6 * Dj_data[:, 1], '--C3', label='$D_j$')  # plot Dj
ax11.plot(E, 1e6 * (C[0] * v0 * ((nB+1)*rho_em + nB*rho_ab) + Dj_data[:, 1]), '-C0',
         label='$C =$ ' + str(C[0]))  # plot the sum
ax11.plot(E, 1e6 * (C[1] * v0 * ((nB+1)*rho_em + nB*rho_ab) + Dj_data[:, 1]), '-C4',
         label='$C =$ ' + str(C[1]))  # plot the sum

# configuring the plot
ax1.set_xlim(min(E), max(E))
ax1.set_ylim(1e-4, 10)
ax1.set_yscale('log')
ax1.set_xlabel('eV', fontsize=16)
ax1.set_ylabel(r'$10^{18}{\rm eV}^{-1}{\rm cm}^{-3}$', fontsize=16)
ax1.tick_params(labelsize=16)
ax1.legend(frameon=False, fontsize=15)
# the inset
ax11.set_xlim(0.4, 0.5)
ax11.set_ylim(0, 0.02)

ax2.set_xlim(min(E), max(E))
ax2.set_ylim(0, 7)
ax2.set_yscale('linear')
ax2.set_xlabel('eV', fontsize=16)
ax2.set_ylabel(r'$10^{18}{\rm eV}^{-1}{\rm cm}^{-3}$', fontsize=16)
ax2.tick_params(labelsize=16)
# ax2.legend(frameon=False, fontsize=15)

plt.show()
