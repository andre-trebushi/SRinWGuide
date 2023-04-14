#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:20:42 2023

@author: trebushi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:27:36 2023

@author: trebushi
"""


import time
import plotly.graph_objects as go
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from SRinWGuide import *
import xraylib
import math
import logging
import matplotlib
import logging

import ocelot
from ocelot.common.globals import *
from ocelot.optics.wave import *
from ocelot.gui.dfl_plot import *
from ocelot import ocelog
from ocelot.cpbd import *
from ocelot.rad import *

filePath = '/Users/trebushi/Documents/XFEL/project_THz/poster/'
fileName = 'field trajectory'
# ebeam parameters
beam = Beam()
beam.E = 4.0            # beam energy in [GeV]
beam.I = 0.4 
gamma = beam.E/0.51099890221e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)

# undulator parameters
B0 = 2
lperiod = 0.9# [m] undulator period 
nperiods = 1.5
L_w = lperiod * nperiods
K = 0.9336 * B0 * lperiod * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 

und = Undulator(Kx=2*K, nperiods=nperiods, lperiod=lperiod, eid="und", end_poles='3/4')

xlamds = (lperiod/2/gamma**2)*(1 + K**2/2)
E_ph = speed_of_light * h_eV_s / xlamds
THz = E_ph2THz(E_ph)

lat = MagneticLattice(und)
motion = track2motion(beam, lat, energy_loss=0, quantum_diff=0, accuracy=1)
plot_motion(motion, filePath=filePath + fileName,  savefig=0)
#%%


plt.figure()
plt.plot(motion.z, motion.x)
plt.show()
#%%


x0, y0 = np.meshgrid(motion.x, motion.y)
phi0 = np.arctan2(y0, x0)
r0 = np.sqrt(x0**2 + y0**2)



R = 0.02
m = 0
k = 1001
mu_mk_list = special.jnp_zeros(m, k) #/ 2 / np.pi
mu_mk = mu_mk_list[500]

x1 = np.linspace(-10, 10, 1000)
B = special.jv(m-1, mu_mk*motion.x/R ) #* np.cos((m-1) * phi0)

plt.figure()
# plt.plot(x1, B)
plt.plot(motion.x, B)
# plt.ylim(0, -1)
# plt.plot(B[:, x0.shape[0]//2])

plt.show()

# print((speed_of_light * mu_mk**2/ 2 / R * h_eV_s / 2 / np.pi) * R/L_w / 25)

print((speed_of_light * L_w / 2 / R*2 * mu_mk  * (h_eV_s / 2 / np.pi)))

# print(R/L_w)
#%%


import numpy as np
from scipy.special import comb

def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result

import matplotlib.pyplot as plt

x = np.linspace(-0.5, 100, 1000)

y = smoothstep((x -0.4)*0.01, N=1)
plt.plot(x, y, label=str(100))

plt.legend()

# S((omega - (speed_of_light * mu_mk**2/ 2 / R ) * R/1.35/100) / (0.002 / h_eV_s * 2 * np.pi), N=1)





