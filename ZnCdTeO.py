import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker
from HMAfunctions import mu_n_sharp0, wp, wp0

# y Cadmium fraction
# x Oxygen fraction
# BAC parameters
VZn = 2.8  # eV [Tooru Tanaka et al 2016 Appl. Phys. Express 9 021202]
VCd = 2.2  # eV [Tooru Tanaka et al 2016 Appl. Phys. Express 9 021202]
mZn = 0.117  # me [Adachi (Table 7.1)]
mCd = 0.09  # me [Adachi (Table 7.1)]
EOZn = 1.98  # eV [Tooru Tanaka et al 2016 Appl. Phys. Express 9 021202]
EOCd = 1.88  # eV [Tooru Tanaka et al 2016 Appl. Phys. Express 9 021202] + 
		#  [M Wena et al 2015 Semicond. Sci. Technol. 30 085018 (Fig. 1)]
EgZn = 2.25  # eV [Tooru Tanaka et al 2016 Appl. Phys. Express 9 021202 (almost the same as Adachi)]
EgCd = 1.5  # eV [Tooru Tanaka et al 2016 Appl. Phys. Express 9 021202 (same as Adachi)]
C = 0.46  # eV the bowing factor [Tooru Tanaka et al 2016 Appl. Phys. Express 9 021202]


# the coupling factor
def v(y): return y * VCd + (1 - y) * VZn


# the effective mass
def m(y): return y * mCd + (1 - y) * mZn


# the Oxygen level
def eo(y): return y * EOCd + (1 - y) * EOZn


# the band gap
def eg(y): return y * EgCd + (1 - y) * EgZn - C * y * (1 - y)


# Ed
def ed(y): return eo(y) - eg(y)


x0 = 0.012
yy = 0.
l = 10

Y = np.linspace(0, 1, 100)
plt.plot(Y, eg(Y))
plt.plot(Y, eo(Y))
plt.axvline(x=0.27446, ls='--', color='k')

plt.figure()
plt.plot(Y, ed(Y), lw=3)
plt.xlim(0, 1)
plt.xlabel('y', fontsize=16)
plt.ylabel('$E_d$ [eV]', fontsize=16)
plt.axhline(y=0, ls='--', color='k')
plt.axvline(x=0.27446, ls='--', color='k')
plt.text(0.22, 0.1, 'y = 0.27', rotation=90, fontsize=16)
plt.tick_params(labelsize=16)

print('Ed(0) =', ed(0))
print('Ed(1) =', ed(1))

'''
n = np.logspace(-9, -5, 200)
Mu = np.zeros(len(n))
Wp = np.zeros(len(n))
for i in range(len(n)):
    Mu[i] = mu_n_sharp0(n[i], ed(yy), v(yy), x0, m(yy))
    if Mu[i] == ed(yy):
        Wp[i] = float('nan')
    else:
        Wp[i] = wp(Mu[i], ed(yy), l, v(yy), x0, m(yy))

plt.figure()
plt.plot(1e24*n, 1e3*Wp)
plt.xscale('log')
plt.xlim(1e15, 1e19)
plt.ylim(0, 40)
plt.xlabel('n [cm$^{-3}$]')
plt.ylabel('$\hbar\omega_p$ [meV]')
plt.title('Zn$_{1-y}$Cd$_y$Te$_{1-x}$O$_x$, y = ' + str(yy) + ', x = ' + str(x0) + ', l = 10 $\AA$')



n = np.logspace(-9, -5, 200)
nn, YY = np.meshgrid(n, Y)

Mu = np.zeros([len(Y), len(n)])
Wp0 = np.zeros([len(Y), len(n)])
Wp = np.zeros([len(Y), len(n)])
for j in range(len(Y)):
    for i in range(len(n)):
        Mu[j, i] = mu_n_sharp0(n[i], ed(Y[j]), v(Y[j]), x0, m(Y[j]))
        Wp0[j, i] = wp0(Mu[j, i], ed(Y[j]), v(Y[j]), x0, m(Y[j]))
        if Mu[j, i] == ed(Y[j]):
            Wp[j, i] = float('nan')
        else:
            Wp[j, i] = wp(Mu[j, i], ed(Y[j]), l, v(Y[j]), x0, m(Y[j]))
    print(j)

fig1, ax1 = plt.subplots()
# fig1.subplots_adjust(bottom=0.13)
ax2 = ax1.twiny()

CS1 = ax1.imshow(1e3 * Wp, aspect='auto', origin='lower', extent=[-9, -5, 0, 1])
ax1.set_xticks([])
cbar = plt.colorbar(CS1)
cbar.set_label('$\hbar\omega_p$ [meV]')

CS2 = ax2.contour(1e24 * n, Y, 1e3 * Wp, linewidths=1, colors='C1',
                  levels=15)
ax2.clabel(CS2, colors='C3', fmt='%1.0f')
ax2.set_axisbelow(False)
ax2.set_xscale('log')
ax2.xaxis.tick_bottom()
ax2.set_xlabel('n [cm$^{-3}$]')
ax2.xaxis.set_label_position('bottom')
ax1.set_ylabel('y')

plt.title('Zn$_{1-y}$Cd$_y$Te$_{1-x}$O$_x$ , x = ' + str(x0) + ', l = 10 $\AA$')

print(1e3 * np.nanmin(Wp))
print(1e3 * np.nanmax(Wp))

'''
plt.show()
