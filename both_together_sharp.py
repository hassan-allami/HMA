import numpy as np
from matplotlib import pyplot as plt

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

x0 = 0.012  # the alloy fraction
v0 = 6.009 ** 3  # unit cell volume for ZeTe
C = 100  # the coefficient
n = [1, 3, 5, 7, 9]  # set of density levels

# form figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.subplots_adjust(wspace=0.3, left=0.1, right=0.96)
fig.suptitle('ZnTe$_{0.988}$O$_{0.012}$, $n_0 = 10^{16}$ cm$^{-3}$, $C =$ '+str(C), fontsize=18)

for j in range(5):
    # open files
    f_rho = open('Data/rhoj_ZnTeO'+str(x0)+'_'+str(n[j])+'n016.txt')  # open file for rhoj
    f_D = open('Data/Dj_ZnTeO'+str(x0)+'_'+str(n[j])+'n016.txt')  # open file for Dj
    # read data for rhoj
    rho_data = f_rho.read()
    rho_data = rho_data.split()
    rho_data = np.array(list(map(float, rho_data)))
    rho_data = np.reshape(rho_data, (int(len(rho_data) / 2), 2))
    # read data for Dj
    Dj_data = f_D.read()
    Dj_data = Dj_data.split()
    Dj_data = np.array(list(map(float, Dj_data)))
    Dj_data = np.reshape(Dj_data, (int(len(Dj_data) / 2), 2))
    # close the files
    f_rho.close()
    f_D.close()
    # plot
    ax1.plot(rho_data[:, 0], 1e4 * C*v0*rho_data[:, 1], ':C'+str(j))  # plot rhoj
    ax1.plot(Dj_data[:, 0], 1e4 * Dj_data[:, 1], '--C' + str(j))  # plot Dj
    ax1.plot(rho_data[:, 0], 1e4 * (C * v0 * rho_data[:, 1] + Dj_data[:, 1]), 'C' + str(j),
             label='$n =$ '+str(n[j])+'$n_0$')  # plot the sum
    ax2.plot(rho_data[:, 0], 1e4 * C * v0 * rho_data[:, 1], ':C' + str(j))  # plot rhoj
    ax2.plot(Dj_data[:, 0], 1e4 * Dj_data[:, 1], '--C' + str(j))  # plot Dj
    ax2.plot(rho_data[:, 0], 1e4 * (C * v0 * rho_data[:, 1] + Dj_data[:, 1]), 'C' + str(j),
             label='$n =$ ' + str(n[j]) + '$n_0$')  # plot the sum

# configuring the plot
ax1.set_xlim(min(rho_data[:, 0]), max(rho_data[:, 0]))
ax1.set_ylim(1e-8, 0.01)
ax1.set_yscale('log')
ax1.set_xlabel('eV', fontsize=16)
ax1.set_ylabel(r'$10^{20}{\rm eV}^{-1}{\rm cm}^{-3}$', fontsize=16)
ax1.tick_params(labelsize=16)

ax2.set_xlim(min(rho_data[:, 0]), max(rho_data[:, 0]))
ax2.set_ylim(0, 0.01)
ax2.set_yscale('linear')
ax2.set_xlabel('eV', fontsize=16)
ax2.set_ylabel(r'$10^{20}{\rm eV}^{-1}{\rm cm}^{-3}$', fontsize=16)
ax2.tick_params(labelsize=16)
ax2.legend(frameon=False, fontsize=15)

plt.show()
