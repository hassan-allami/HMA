import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy import optimize
from scipy.special import digamma

cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in ev^(-3/2)A^(-3)
e2 = 1973 / 137  # e^2 = \hbar * c * alapha in eV.A
kB = 8.617 * 1e-5  # Boltzmann constant in ev/Kelvin
m0 = 0.1  # in me
V = 2.8  # in eV
x0 = 0.01  # the alloy fraction


''' the general bands and weights '''


def epp(k, ed, g, v=V, x=x0):  # returns E_+^+
    return (np.complex(k ** 2 + ed, g) + np.sqrt(np.complex(k ** 2 - ed, -g) ** 2 + 4 * v ** 2 * x)) / 2


def epm(k, ed, g, v=V, x=x0):  # returns E_+^-
    return (np.complex(k ** 2 + ed, -g) + np.sqrt(np.complex(k ** 2 - ed, g) ** 2 + 4 * v ** 2 * x)) / 2


def emp(k, ed, g, v=V, x=x0):  # returns E_-^+
    return (np.complex(k ** 2 + ed, g) - np.sqrt(np.complex(k ** 2 - ed, -g) ** 2 + 4 * v ** 2 * x)) / 2


def emm(k, ed, g, v=V, x=x0):  # returns E_-^+
    return (np.complex(k ** 2 + ed, -g) - np.sqrt(np.complex(k ** 2 - ed, g) ** 2 + 4 * v ** 2 * x)) / 2


# the general weightings
def app(k, ed, g, v=V, x=x0):  # returns a_+^+
    return v ** 2 * x / (epp(k, ed, g, v, x) - emp(k, ed, g, v, x)) / (epp(k, ed, g, v, x) - k ** 2 + 1e-13)


def apm(k, ed, g, v=V, x=x0):  # returns a_+^-
    return v ** 2 * x / (epm(k, ed, g, v, x) - emm(k, ed, g, v, x)) / (epm(k, ed, g, v, x) - k ** 2)


def amp(k, ed, g, v=V, x=x0):  # returns a_-^+
    return v ** 2 * x / (epp(k, ed, g, v, x) - emp(k, ed, g, v, x)) / (k ** 2 - emp(k, ed, g, v, x))


def amm(k, ed, g, v=V, x=x0):  # returns a_-^-
    return v ** 2 * x / (epm(k, ed, g, v, x) - emm(k, ed, g, v, x)) / (k ** 2 - emm(k, ed, g, v, x))


''' the END of general bands and weights '''


# define the integrand = Im[a+a- / E - E+ + E-]
def integrand(k, ed, gamma, e, v=V, x=x0):
    denom1 = np.complex(e, -gamma) - np.real(np.sqrt(4 * v ** 2 * x + np.complex(k ** 2 - ed, -gamma) ** 2))
    denom2 = np.complex(k ** 2 - ed, gamma) + np.sqrt(4 * v ** 2 * x + np.complex(k ** 2 - ed, gamma) ** 2)
    denom3 = np.complex(-k ** 2 + ed, gamma) + np.sqrt(4 * v ** 2 * x + np.complex(k ** 2 - ed, -gamma) ** 2)
    denom4 = np.abs(4 * v ** 2 * x + np.complex(k ** 2 - ed, -gamma) ** 2)
    return 4 * v ** 4 * x ** 2 * k ** 2 * np.imag(1 / (denom1 * denom2 * denom3)) / denom4 / np.pi ** 3


# perform the 1D k-integral to find Dj with finite Gamma
def jdos(e, ed, g, v=V, x=x0, m=m0):
    result, err = quad(integrand, 0, np.inf, args=(ed, g, e, v, x))
    if err / result < 1e-3:
        dj = result * cte * m ** (3 / 2)
    else:
        print(err / result)
        dj = float('nan')
    return dj


# define Dj for Gamma = 0
def jdos_sharp(e, ed, v=V, x=x0, m=m0):
    if e > np.sqrt(ed ** 2 + 4 * v ** 2 * x):
        dj = np.sqrt(ed + np.sqrt(e ** 2 - 4 * v ** 2 * x)) / (e * np.sqrt(e ** 2 - 4 * v ** 2 * x))
    elif ed < 0:
        dj = 0
    elif e > 2 * v * np.sqrt(x):
        dj = (np.sqrt(ed + np.sqrt(e ** 2 - 4 * v ** 2 * x)) + np.sqrt(ed - np.sqrt(e ** 2 - 4 * v ** 2 * x))) / \
             (e * np.sqrt(e ** 2 - 4 * v ** 2 * x))
    else:
        dj = 0
    return cte * m ** (3 / 2) * v ** 2 * x * dj / (2 * np.pi ** 2)


# define broadened Gamma with eta at Ed
def gamma(ed, eta, m=m0):
    v = 5.6 ** 3  # unit cell volume
    beta = 0.2  # the constant pre-factor
    return cte * m ** (3 / 2) * v * beta * np.real(np.sqrt(np.complex(ed, eta))) / (2 * np.pi)


# finding max(Dj) for a given Ed
def dj_max(ed, eta, v=V, x=x0, m=m0):
    if ed < -0.05:
        E = np.linspace(np.sqrt(ed ** 2 + 4 * v ** 2 * x), 1.3 * np.sqrt(ed ** 2 + 4 * v ** 2 * x), 200)
        Dj = np.zeros(len(E))
        for i in range(len(E)):
            Dj[i] = jdos_sharp(E[i], ed, m)
    else:
        E = np.linspace(2 * v * np.sqrt(x), 1.03 * 2 * v * np.sqrt(x), 200)
        Dj = np.zeros(len(E))
        for i in range(len(E)):
            Dj[i] = jdos(E[i], ed, gamma(ed, eta, m), m)

    index = np.asarray(Dj == max(Dj)).nonzero()[0]
    if index.size == 0:
        print('ooops!')
        emax = float('nan')
    else:
        emax = E[index[0]]
    return emax, max(Dj)


''' for indirect transitions '''


# DOS of sharp Ep
def rhop_sharp(e, ed, v=V, x=x0, m=m0):
    ep0 = (ed + np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    if e > ep0:
        rho = np.sqrt(ed + ((e - ed) ** 2 - v ** 2 * x) / (e - ed))
    else:
        rho = 0
    return cte * m ** (3 / 2) * rho / (2 * np.pi ** 2)


# DOS of sharp Em
def rhom_sharp(e, ed, v=V, x=x0, m=m0):
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    if em0 < e < ed:
        rho = np.sqrt(ed + ((e - ed) ** 2 - v ** 2 * x) / (e - ed))
    else:
        rho = 0
    return cte * m ** (3 / 2) * rho / (2 * np.pi ** 2)


# integrand for Ep DOS
def int_rhop(k, ed, g, e, v=V, x=x0):
    return k ** 2 * np.imag(app(k, ed, g, v, x) / (e - epp(k, ed, g, v, x))) / np.pi ** 3


# integrand for Em DOS
def int_rhom(k, ed, g, e, v=V, x=x0):
    epp = (np.complex(k ** 2 + ed, g) + np.sqrt(np.complex(k ** 2 - ed, -g) ** 2 + 4 * v ** 2 * x)) / 2
    emp = (np.complex(k ** 2 + ed, g) - np.sqrt(np.complex(k ** 2 - ed, -g) ** 2 + 4 * v ** 2 * x)) / 2
    return k ** 2 * np.imag(v ** 2 * x / (epp - emp) / (k ** 2 - emp) / (e - emp)) / np.pi ** 3


# perform the 1D k-integral to find rho_p with finite Gamma
def rhop(e, ed, g, v=V, x=x0, m=m0, upper=np.inf):
    output = quad(int_rhop, 0, upper, args=(ed, g, e, v, x), full_output=1)
    if len(output) == 3:
        rho = output[0] * cte * m ** (3 / 2)
    else:
        # print('rhop relative error at e =', e, 'was ', np.abs(err / result), ', so rhop_sharp was used instead')
        rho = rhop_sharp(e, ed, v, x, m)
    return rho


# perform the 1D k-integral to find rho_m with finite Gamma
def rhom(e, ed, g, v=V, x=x0, m=m0, upper=np.inf):
    result, err = quad(int_rhom, 0, upper, args=(ed, g, e, v, x))
    if np.abs(err / result) < 1e-2:
        rho = result * cte * m ** (3 / 2)
    else:
        # print('rhom relative error at e =', e, 'was ', np.abs(err / result), ', so rhom_sharp was used instead')
        rho = rhom_sharp(e, ed, v, x, m)
    return rho


# integrand for sharp rho_j at T = 0
def int_rhoj_sharp(e1, ed, omega, v=V, x=x0):
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    ep0 = (ed + np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    return np.sqrt((e1 - em0) * (e1 - ep0) * (e1 + omega - em0) * (e1 + omega - ep0) / (e1 - ed) / (e1 + omega - ed))


# solve sharp rho_j integral at T = 0
def rhoj_sharp(omega, ed, mu, v=V, x=x0, m=m0):
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    ep0 = (ed + np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    el = max(em0, ep0 - omega)
    eu = min(ed, mu)
    if eu > el:
        result, err = quad(int_rhoj_sharp, el, eu, args=(ed, omega, v, x))
        if np.abs(err / result) < 1e-3:
            rho = result * cte ** 2 * m ** 3 / (4 * np.pi ** 4)
        else:
            print(err / result)
            rho = float('nan')
    else:
        rho = 0
    return rho


# integrand for sharp rho_j at T = 0
def int_rhoj(e1, ed, g, omega, v=V, x=x0, m=m0):
    return rhom(e1, ed, g, v, x, m, upper=4) * rhop(e1 + omega, ed, g, v, x, m, upper=4)


# solve rho_j integral at T = 0
def rhoj(omega, ed, mu, g, v=V, x=x0, m=m0):
    result, err = quad(int_rhoj, mu - omega, mu, args=(ed, g, omega, v, x, m), epsabs=1e-13)
    if omega > 0:
        if np.abs(err / result) < 1:
            rho = result
        else:
            print('rhoj did not converge at Omega =', omega)
            rho = float("nan")
    else:
        rho = 0
    return rho


'''' with filling '''


# Fermi distribution
def f(e, t):
    if e / (kB * t) < 10:
        result = 1 / (np.exp(e / (kB * t)) + 1)
    else:
        result = 0
    return result


# integrand for max Em density at Gamma = 0 and T = 0
def int_nmax_sharp(k, ed, v=V, x=x0):
    return k ** 2 * (1 - (k ** 2 - ed) / np.sqrt((k ** 2 - ed) ** 2 + 4 * v ** 2 * x))


# define max Em density at Gamma = 0 and T = 0
def nmax_sharp(ed, v=V, x=x0, m=m0, upper=np.inf):
    result, err = quad(int_nmax_sharp, 0, upper, args=(ed, v, x), epsabs=1e-10)
    if np.abs(err / result) < 1:
        nmax = result
    else:
        print('nmax did not converge', err / result)
        nmax = float("nan")
    return cte * m ** (3 / 2) * nmax / (2 * np.pi ** 2)


# integrand for Em density at Gamma = 0 and T = 0
def int_n_sharp0(k, ed, v=V, x=x0):
    return k ** 2 * (1 - (k ** 2 - ed) / np.sqrt((k ** 2 - ed) ** 2 + 4 * v ** 2 * x))


# define Em density at Gamma = 0 and T = 0
def n_sharp0(mu, ed, v=V, x=x0, m=m0):
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    if em0 < mu < ed:
        kf = np.sqrt(mu + v ** 2 * x / (ed - mu))
        result, err = quad(int_n_sharp0, 0, kf, args=(ed, v, x))
        if np.abs(err / result) < 1:
            n = result
        else:
            print('n did not converge', err / result)
            n = float("nan")
    elif mu <= em0:
        n = 0
    else:
        n, err = quad(int_n_sharp0, 0, np.inf, args=(ed, v, x))
    return cte * m ** (3 / 2) * n / (2 * np.pi ** 2)


# integrand for Em density at Gamma = 0
def int_n_sharp(k, ed, mu, t=300, v=V, x=x0):
    a = (1 - (k ** 2 - ed) / np.sqrt((k ** 2 - ed) ** 2 + 4 * v ** 2 * x))
    F = f((k ** 2 + ed - np.sqrt((k ** 2 - ed) ** 2 + 4 * v ** 2 * x)) / 2 - mu, t)
    return k ** 2 * a * F


# define Em density at Gamma = 0
def n_sharp(mu, ed, t=300, v=V, x=x0, m=m0):
    result, err = quad(int_n_sharp, 0, np.inf, args=(ed, mu, t, v, x))
    if np.abs(err / result) < 1:
        n = result
    else:
        print('n did not converge', err / result)
        n = 0
    return cte * m ** (3 / 2) * n / (2 * np.pi ** 2)


# evaluates density for finite Gamma at T = 0
def n_broad0(mu, ed, g, v=V, x=x0, m=m0):
    # define the integrand
    def int_n(e, ed, g, v, x, m):
        return rhom(e, ed, g, v, x, m) + rhop(e, ed, g, v, x, m)

    # perform the E integral
    result, err = quad(int_n, -np.inf, mu, args=(ed, g, v, x, m), epsabs=1e-10)
    if np.abs(err / result) < 1:
        n = result
    else:
        print('n did not converge', err / result)
        n = 0
    return n


# evaluates density for finite Gamma
def n_broad(mu, ed, g, t=300, v=V, x=x0, m=m0):
    # define the integrand
    def int_n(e, mu, ed, g, t, v, x, m):
        return (rhom(e, ed, g, v, x, m) + rhop(e, ed, g, v, x, m, upper=4)) * f(e - mu, t)

    # perform the E integral
    output = quad(int_n, -np.inf, np.inf, args=(mu, ed, g, t, v, x, m), limit=50, full_output=1)
    if len(output) == 3:
        n = output[0]
    else:
        n = n_sharp(mu, ed, t, v, x, m)
    return n


# find mu for a given n and T = 0
def mu_n_sharp0(n, ed, v=V, x=x0, m=m0):
    def eq(mu):
        return n_sharp0(mu, ed, v, x, m) - n

    margin = 0
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    if 0 < n < nmax_sharp(ed, v, x, m):
        sol = optimize.root_scalar(eq, bracket=[em0 - margin, ed + margin], method='brentq')
        root = sol.root
    else:
        print('not in range')
        root = ed
    return root


# find mu for a given n and T
def mu_n_sharp(n, ed, t=300, v=V, x=x0, m=m0):
    def eq(mu):
        return n_sharp(mu, ed, t, v, x, m) - n

    margin = 0.2
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    if 0 < n < nmax_sharp(ed, v, x, m):
        sol = optimize.root_scalar(eq, bracket=[em0 - margin, ed + margin], method='brentq')
        root = sol.root
    else:
        print('not in range')
        root = float('nan')
    return root


# find kF for a given mu
def kf_mu(mu, ed, v=V, x=x0, m=m0):
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2

    def eq(k):
        return k ** 2 + ed - np.sqrt((k ** 2 - ed) ** 2 + 4 * v ** 2 * x) - 2 * mu

    if em0 < mu < ed:
        sol = optimize.root_scalar(eq, bracket=[0, 1e3], method='brentq')
        kf = sol.root
    else:
        print('not in range')
        kf = float('nan')
    return cte ** (1 / 3) * np.sqrt(m) * kf


# find mu for a given n and T
def mu_n_broad(n, ed, g, t=300, v=V, x=x0, m=m0):
    def eq(mu):
        return n_broad(mu, ed, g, t, v, x, m) - n

    margin = 0.2
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    if 0 < n < nmax_sharp(ed, v, x, m, upper=10):
        sol = optimize.root_scalar(eq, bracket=[em0 - margin, ed + margin], method='brentq')
        root = sol.root
    else:
        print('not in range')
        root = float('nan')
    return root


# find mu for a given n and T = 0
def mu_n_broad0(n, ed, g, v=V, x=x0, m=m0):
    def eq(mu):
        return n_broad0(mu, ed, g, v, x, m) - n

    margin = 0.2
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    if 0 < n < nmax_sharp(ed, v, x, m):
        sol = optimize.root_scalar(eq, bracket=[em0 - margin, ed + margin], method='brentq')
        root = sol.root
    else:
        print('not in range')
        root = float('nan')
    return root


# define Dj with filling at Gamma = 0
def jdos_mu_sharp(e, mu, ed, t=300, v=V, x=x0, m=m0):
    if e > np.sqrt(ed ** 2 + 4 * v ** 2 * x):
        dj = np.sqrt(ed + np.sqrt(e ** 2 - 4 * v ** 2 * x)) / (e * np.sqrt(e ** 2 - 4 * v ** 2 * x)) * \
             f(ed - mu + (np.sqrt(e ** 2 - 4 * v ** 2 * x) - e) / 2, t) * \
             f(mu - ed - (np.sqrt(e ** 2 - 4 * v ** 2 * x) + e) / 2, t)
    elif ed < 0:
        dj = 0
    elif e > 2 * v * np.sqrt(x):
        dj = np.sqrt(ed + np.sqrt(e ** 2 - 4 * v ** 2 * x)) / (e * np.sqrt(e ** 2 - 4 * v ** 2 * x)) * \
             f(ed - mu + (np.sqrt(e ** 2 - 4 * v ** 2 * x) - e) / 2, t) * \
             f(mu - ed - (np.sqrt(e ** 2 - 4 * v ** 2 * x) + e) / 2, t) + \
             np.sqrt(ed - np.sqrt(e ** 2 - 4 * v ** 2 * x)) / (e * np.sqrt(e ** 2 - 4 * v ** 2 * x)) * \
             f(ed - mu - (np.sqrt(e ** 2 - 4 * v ** 2 * x) + e) / 2, t) * \
             f(mu - ed + (np.sqrt(e ** 2 - 4 * v ** 2 * x) - e) / 2, t)
    else:
        dj = 0
    return cte * m ** (3 / 2) * v ** 2 * x * dj / (2 * np.pi ** 2)


''' direct with filling and broadening '''


# correction functions C_\pm
def cp(k, ed, g, mu, t, e, v=V, x=x0):  # returns C+
    exp1 = (digamma(1 / 2 + 1j * (emm(k, ed, g, v, x) - mu) / (2 * np.pi * kB * t)) - digamma(
        1 / 2 + 1j * (emm(k, ed, g, v, x) + e - mu) / (2 * np.pi * kB * t))) * \
           (app(k, ed, g, v, x) * amm(k, ed, g, v, x) / (e - epp(k, ed, g, v, x) + emm(k, ed, g, v, x)) -
            apm(k, ed, g, v, x) * amm(k, ed, g, v, x) / (e - epm(k, ed, g, v, x) + emm(k, ed, g, v, x)))
    exp2 = (digamma(1 / 2 + 1j * ((epp(k, ed, g, v, x) - mu) / (2 * np.pi * kB * t))) - digamma(
        1 / 2 + 1j * ((epp(k, ed, g, v, x) - e - mu) / (2 * np.pi * kB * t)))) * \
           app(k, ed, g, v, x) * amm(k, ed, g, v, x) / (e - epp(k, ed, g, v, x) + emm(k, ed, g, v, x))
    exp3 = (digamma(1 / 2 - 1j * ((epp(k, ed, g, v, x) - mu) / (2 * np.pi * kB * t))) - digamma(
        1 / 2 - 1j * ((epp(k, ed, g, v, x) - e - mu) / (2 * np.pi * kB * t)))) * \
           app(k, ed, g, v, x) * amp(k, ed, g, v, x) / (e - epp(k, ed, g, v, x) + emp(k, ed, g, v, x))
    return exp1 + exp2 - exp3


def cm(k, ed, g, mu, t, e, v=V, x=x0):  # returns C-
    exp1 = (digamma(1 / 2 - 1j * (epp(k, ed, g, v, x) - mu) / (2 * np.pi * kB * t)) - digamma(
        1 / 2 - 1j * (epp(k, ed, g, v, x) - e - mu) / (2 * np.pi * kB * t))) * \
           (app(k, ed, g, v, x) * amm(k, ed, g, v, x) / (e - epp(k, ed, g, v, x) + emm(k, ed, g, v, x)) -
            app(k, ed, g, v, x) * amp(k, ed, g, v, x) / (e - epp(k, ed, g, v, x) + emp(k, ed, g, v, x)))
    exp2 = (digamma(1 / 2 - 1j * ((emm(k, ed, g, v, x) - mu) / (2 * np.pi * kB * t))) - digamma(
        1 / 2 - 1j * ((emm(k, ed, g, v, x) + e - mu) / (2 * np.pi * kB * t)))) * \
           app(k, ed, g, v, x) * amm(k, ed, g, v, x) / (e - epp(k, ed, g, v, x) + emm(k, ed, g, v, x))
    exp3 = (digamma(1 / 2 + 1j * ((emm(k, ed, g, v, x) - mu) / (2 * np.pi * kB * t))) - digamma(
        1 / 2 + 1j * ((emm(k, ed, g, v, x) + e - mu) / (2 * np.pi * kB * t)))) * \
           apm(k, ed, g, v, x) * amm(k, ed, g, v, x) / (e - epm(k, ed, g, v, x) + emm(k, ed, g, v, x))
    return exp1 + exp2 - exp3


# integrand for Dj(mu, E), the first form
def int_dj_filling_1(k, ed, g, mu, e, t=300, v=V, x=x0):
    # the main term
    first = (app(k, ed, g, v, x) * amm(k, ed, g, v, x)) / (e - epp(k, ed, g, v, x) + emm(k, ed, g, v, x)) * \
            f(epp(k, ed, g, v, x) - e - mu, t) * f(mu - epp(k, ed, g, v, x), t)
    # the correction term
    correct = cp(k, ed, g, mu, t, e, v, x) / (2 * np.pi * (1 - np.exp(- e / (kB * t))))
    if first.imag + 1*correct.real > 0:
        total = first.imag + correct.real
    else:
        total = first.imag + correct.real
    return k ** 2 * total


# integrand for Dj(mu, E), the second form
def int_dj_filling_2(k, ed, g, mu, e, t=300, v=V, x=x0):
    # the main term
    first = (app(k, ed, g, v, x) * amm(k, ed, g, v, x)) / (e - epp(k, ed, g, v, x) + emm(k, ed, g, v, x)) * \
            f(emm(k, ed, g, v, x) - mu, t) * f(mu - e - emm(k, ed, g, v, x), t)
    # the correction term
    correct = cm(k, ed, g, mu, t, e, v, x) / (2 * np.pi * (1 - np.exp(- e / (kB * t))))
    if first.imag + 1 * correct.real > 0:
        total = first.imag + 1*correct.real
    else:
        total = first.imag + 1*correct.real
    return k ** 2 * total


# integrand for Dj(mu, E), an approximate form
def int_dj_filling_approx(k, ed, g, mu, e, t=300, v=V, x=x0):
    # the main term
    first = (app(k, ed, g, v, x) * amm(k, ed, g, v, x)) / (e - epp(k, ed, g, v, x) + emm(k, ed, g, v, x))
    second = f(emm(k, ed, g, v, x) - mu, t) * f(mu - epp(k, ed, g, v, x), t)
    if first.imag * second.real > 0:
        total = first.imag * second.real
    else:
        total = 0
    return k ** 2 * total


# evaluate Dj(mu, E) integral, the first form
def dj_filling_1(e, mu, ed, g, t=300, v=V, x=x0, m=m0, upper=4):
    result, err = quad(int_dj_filling_1, 0, upper, args=(ed, g, mu, e, t, v, x), epsabs=1e-10)
    if np.abs(err / result) < 1:
        dj = result
    else:
        print('Dj did not converge', result)
        dj = float('nan')
    return cte * m ** (3 / 2) * dj / np.pi ** 3


# evaluate Dj(mu, E) integral, the second form
def dj_filling_2(e, mu, ed, g, t=300, v=V, x=x0, m=m0, upper=4):
    result, err = quad(int_dj_filling_2, 0, upper, args=(ed, g, mu, e, t, v, x), epsabs=1e-10)
    if np.abs(err / result) < 1:
        dj = result
    else:
        print('Dj did not converge', result)
        dj = float('nan')
    return cte * m ** (3 / 2) * dj / np.pi ** 3


# evaluate Dj(mu, E) integral, the approx form (which is actually less problematic)
def dj_filling_approx(e, mu, ed, g, t=300, v=V, x=x0, m=m0, upper=4):
    result, err = quad(int_dj_filling_approx, 0, upper, args=(ed, g, mu, e, t, v, x), epsabs=1e-10)
    if np.abs(err / (result + 1e-13)) < 1:
        dj = result
    else:
        print('Dj did not converge', result)
        dj = float('nan')
    return cte * m ** (3 / 2) * dj / np.pi ** 3


''' indirect with filling at finite T '''


# integrand for sharp rho_j at finite T
def int_rhoj_sharp_t(e1, ed, omega, mu, t=300, v=V, x=x0):
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    ep0 = (ed + np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    filling = f(e1 - mu, t) * f(mu - omega - e1, t)
    return filling * np.sqrt((e1 - em0) * (e1 - ep0) * (e1 + omega - em0) * (e1 + omega - ep0) /
                             (e1 - ed) / (e1 + omega - ed))


# solve sharp rho_j integral at finite T
def rhoj_sharp_t(omega, ed, mu, t=300, v=V, x=x0, m=m0):
    em0 = (ed - np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    ep0 = (ed + np.sqrt(ed ** 2 + 4 * v ** 2 * x)) / 2
    el = max(em0, ep0 - omega)
    if ed > el:
        result, err = quad(int_rhoj_sharp_t, el, ed, args=(ed, omega, mu, t, v, x))
        if np.abs(err / result) < 1:
            rho = result * cte ** 2 * m ** 3 / (4 * np.pi ** 4)
        else:
            print(err / result)
            rho = float('nan')
    else:
        rho = 0
    return rho


# integrand for sharp rho_j at finite T
def int_rhoj_t(e1, ed, g, omega, mu, t=300, v=V, x=x0, m=m0):
    return rhom(e1, ed, g, v, x, m, upper=4) * rhop(e1 + omega, ed, g, v, x, m, upper=4) * \
           f(e1 - mu, t) * f(mu - omega - e1, t)


# solve rho_j integral
def rhoj_t(omega, ed, mu, g, t=300, v=V, x=x0, m=m0, low=-np.inf, up=np.inf):
    result, err = quad(int_rhoj_t, low, up, args=(ed, g, omega, mu, t, v, x, m), epsabs=1e-13)
    if np.abs(err / result) < 1:
        rho = result
    else:
        print('rhoj_t did not converge', err / result)
        rho = float("nan")
    return rho


''' plasma frequency '''


# define hbar*wp0 in eV
def wp0(mu, ed, v=V, x=x0, m=m0):
    kf = np.sqrt(mu + v ** 2 * x / (ed - mu))  # find kF (in eV^(1/2))
    expr = (kf / 2 * (1 - (kf ** 2 - ed) / np.sqrt((kf ** 2 - ed) ** 2 + 4 * v ** 2 * x))) ** 3  # the bracket
    return np.sqrt(8 * e2 * cte ** (1 / 3) * np.sqrt(m) * expr / (3 * np.pi))


# Eg the energy gap
def energy_gap(mu, ed, v=V, x=x0):
    kf = np.sqrt(mu + v ** 2 * x / (ed - mu))  # find kF (in eV^(1/2))
    if ed <= 0:
        Eg = np.sqrt(ed ** 2 + 4 * v ** 2 * x)
    elif kf ** 2 < ed:
        Eg = np.sqrt((ed - kf ** 2) ** 2 + 4 * v ** 2 * x)
    else:
        Eg = 2 * v * np.sqrt(x)
    return Eg


# define epsilon_cross
def eps_cross(w, mu, ed, l, v=V, x=x0, m=m0):
    kf = np.sqrt(mu + v ** 2 * x / (ed - mu))  # find kF (in eV^(1/2))
    integ = lambda k: k ** 2 / (
            np.sqrt((k ** 2 - ed) ** 2 + 4 * v ** 2 * x) * ((k ** 2 - ed) ** 2 + 4 * v ** 2 * x - w ** 2))
    result, err = quad(integ, 0, kf)
    if np.abs(err / result) < 1e-3:
        eps = 1 + 8 * e2 * cte * m ** (3 / 2) * l ** 2 * v ** 2 * x * result / np.pi
    else:
        result = 1e2 * kf ** 3 / ((kf ** 2 - ed) ** 2 + 4 * v ** 2 * x) ** (3 / 2)
        eps = 1 + 8 * e2 * cte * m ** (3 / 2) * l ** 2 * v ** 2 * x * result / np.pi
    return eps


# solving for wp
def wp(mu, ed, l, v=V, x=x0, m=m0):
    [wl, wr] = [0, energy_gap(mu, ed, v, x) / (1 + 1e-9)]
    wm = (wl + wr) / 2
    n = 1
    while wr - wl > 1e-5 * energy_gap(mu, ed, v, x) and n < 100:
        diff_m = wm ** 2 - wp0(mu, ed, v, x, m) ** 2 / eps_cross(wm, mu, ed, l, v, x, m)
        if diff_m > 0:
            wr = wm
            wm = (wl + wr) / 2
        else:
            wl = wm
            wm = (wl + wr) / 2

        n += 1
    return wm


'''
# fixed parameters
m0 = 0.09  # in me
V = 2.2  # in eV
x0 = 0.01  # the alloy fraction

# other parameters
n0 = 1e-6  # A^(-3) = x 10^24 cm^(-3)
Ed = 0.38  # eV
Gamma = 10e-3  # 10 meV

Em0 = (Ed - np.sqrt(Ed**2 + 4*V**2*x0)) / 2  # E-(0)
print('Em0 =', Em0)
mu = -0.
mu = mu_n_broad(5*n0, Ed, Gamma, v=V, x=x0, m=m0)  # finding mu


Mu = np.linspace(-0.1, Ed, 100)
N = np.zeros(len(Mu))
N0 = np.zeros(len(Mu))
for i in range(len(Mu)):
    N[i] = n_broad(Mu[i], Ed, Gamma, t=300, v=V, x=x0, m=m0)
    N0[i] = n_sharp(Mu[i], Ed, t=300, v=V, x=x0, m=m0)
    print(i)

plt.plot(Mu, N)
plt.plot(Mu, N0)


print('mu =', mu)


Omega = 0.5  # test

Dj1 = dj_filling_1(Omega, mu, Ed, Gamma,  v=V, x=x0, m=m0, upper=10)  # calculate Dj
Dj2 = dj_filling_2(Omega, mu, Ed, Gamma,  v=V, x=x0, m=m0)  # calculate Dj
Dj_approx = dj_filling_approx(Omega, 0.1, Ed, Gamma,  v=V, x=x0, m=m0)  # calculate Dj
print('Dj1 =', Dj1)
print('Dj2 =', Dj2)
print('Dj_approx =', Dj_approx)
print('Delta Dj =', Dj2 - Dj1)


k = np.linspace(0, 5, 1000)
Integrand_1 = np.zeros(len(k))
Integrand_2 = np.zeros(len(k))
Integrand_approx = np.zeros(len(k))
for i in range(len(k)):
    Integrand_1[i] = int_dj_filling_1(k[i], Ed, Gamma, mu, Omega, t=300, v=V, x=x0)
    Integrand_2[i] = int_dj_filling_2(k[i], Ed, Gamma, mu, Omega, t=300, v=V, x=x0)
    Integrand_approx[i] = int_dj_filling_approx(k[i], Ed, Gamma, mu, Omega, t=300, v=V, x=x0)

plt.figure()
plt.plot(k, Integrand_1)
plt.plot(k, Integrand_2)
plt.plot(k, Integrand_approx)
plt.show()
'''
