"""
                    parameter list to share between files
"""
import matplotlib.pyplot as plt

        ### Physical Constants ###
hbar =6.58211899*10**(-16)  # h/2Pi  in eV s *)
m0=9.10938215*10**(-31)
e0=1.602176487*10**(-19)    # electron charge in C
hbm0 = (hbar)**(2) * e0 * 10**(20)/m0   # hbar^2 / m0 in eV * A^2
kb = 8.61733 * 10**(-5) * 1000          # boltzmann constant in meV / K
epsilon_knot = 8.854187 * 10**(-25)     # vacuum permitivity in C/mV*A
eep0 = e0/epsilon_knot  #e/epsilon_knot in mV*A
muB = 5.7883818012 * 10**(-5)       # Bohr magneton in eV * T^-1
phi_knot = 2.067833848 * 10**(5)            # magnetic flux quantum in T * A^2

width = 3.847  #* 3.
height = width * .65
params = {
   'text.usetex': False,
   'axes.labelsize': 10,
   'axes.linewidth': 1.7,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   #"font.family": "serif",
   #'font.serif'  : 'cm',
   #"font.sans-serif": ["Helvetica"],
   #'figure.figsize': (width,height),
   #'mathtext.fontset' : 'custom',
   #'mathtext.default': 'regular',
   'xtick.direction': 'in',
   'ytick.direction': 'in',
   }

plt.rcParams.update(params)
plt.rcParams["mathtext.default"]  = 'regular'
plt.rcParams["figure.figsize"] = (width,height)
plt.rcParams['axes.linewidth'] = 1.
