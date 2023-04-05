#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:01:36 2022

@author: trebushi
"""

import logging
import ocelot
from ocelot.common.globals import *
from ocelot.optics.wave import RadiationField, dfl_waistscan
from ocelot.optics.wave import imitate_sase_dfl, wigner_dfl, dfl_waistscan, generate_gaussian_dfl, dfl_ap_rect, dfl_ap_circ, dfl_interp, wigner_dfl, wigner_smear, dfl_chirp_freq
from ocelot.optics.wave import dfl_reflect_surface
from ocelot.gui.dfl_plot import plot_dfl, plot_wigner, plot_dfl_waistscan
from copy import deepcopy
from ocelot import ocelog
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ocelot.optics.wave import *

from ocelot.common.globals import *
from ocelot.common.math_op import find_nearest_idx, fwhm, std_moment, bin_scale, bin_array, mut_coh_func
from ocelot.common.py_func import filename_from_path

from tqdm import tqdm

import xraylib
import math
import logging
import matplotlib

THz = 3
lambds = 0.000299792458 / THz
sigma_r = 55e-3#np.sqrt(lambds * 10)/4/np.pi
print(sigma_r/lambds)
print(lambds)
kwargs={'xlamds':(lambds), #[m] - central wavelength
        'shape':(351, 351, 1),           #(x,y,z) shape of field matrix (reversed) to dfl.fld
        'dgrid':(sigma_r*12,sigma_r*12, 10), #(x,y,z) [m] - size of field matrix
        'power_rms':(sigma_r, sigma_r, 1),#(x,y,z) [m] - rms size of the radiation distribution (gaussian)
        'power_center':(0, 0, None),     #(x,y,z) [m] - position of the radiation distribution
        'power_angle':(0, 0),           #(x,y) [rad] - angle of further radiation propagation
        'power_waistpos':(0, 0),     #(Z_x,Z_y) [m] downstrean location of the waist of the beam
        'wavelength':None,             #central frequency of the radiation, if different from xlamds
        'zsep':None,                   #distance between slices in z as zsep*xlamds
        'freq_chirp':0,                #dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
        'en_pulse':None,               #total energy or max power of the pulse, use only one
        'power':1e6,
        }
dfl = generate_gaussian_dfl(**kwargs);
# plot_dfl(dfl,fig_name='dfl at source domain', domains = "fs")
# plot_dfl(dfl, fig_name='dfl at source k domain', domains = "fk")
#%%
dfl.to_domain('kf')
k_x, k_y = np.meshgrid(dfl.scale_kx(), dfl.scale_ky())
# k = dfl.scale_kz()

dk = 2 * pi / dfl.Lz()
k = 2 * pi / dfl.xlamds
k = np.linspace(k - dk / 2 * dfl.Nz(), k + dk / 2 * dfl.Nz(), dfl.Nz())
print('k_0 =', k[0])
print('k_x =',np.max(k_x))
# print(np.sqrt(k[0] ** 2 - k_x ** 2 - k_y ** 2) - k[0])
dfl_prop = deepcopy(dfl)

#%%

# dfl_prop.prop_m(z=1, m=1.5)
# plot_dfl(dfl_prop,fig_name='dfl at exit of the undulator', domains = "fs")
# plot_dfl(dfl_prop, fig_name='dfl at exit of the undulator', domains = "fk")

#%%
# print(r**2/(lambds*z))

N = 1666
z = 0.3
r = 55e-3

L = N*z

n = 20000
dl = L/n

dfl_waveguide = deepcopy(dfl)
j=0
dfl_waveguide.to_domain('sf')

I_x_array=np.zeros((dfl_waveguide.Nx(), n))
I_y_array=np.zeros((dfl_waveguide.Ny(), n))

loss_array = np.array([])
for i in range(n):
    if (dl*i > j*z):
        P_before=dfl_waveguide.E()
        dfl_waveguide = dfl_ap_circ(dfl_waveguide, r=r)
        P_after=dfl_waveguide.E()
        loss_array = np.append(loss_array, round((1-((P_after)/P_before))*100, 2))
        j = j+1
        print('here')
        
    dfl_waveguide.prop_m(z=dl, m=1)
    # dfl_waveguide = dfl_ap_circ(dfl_waveguide, r=0.95*dfl_waveguide.Lx()/2)

    dfl_waveguide.to_domain('sf')
    I_x_array[:, i] = dfl_waveguide.intensity()[0,dfl_waveguide.Ny()//2,:]
    I_y_array[:, i] = dfl_waveguide.intensity()[0,:,dfl_waveguide.Nx()//2]

#%%

dfl_waveguide2 = deepcopy(dfl)
I_x_array2=np.zeros((dfl_waveguide2.Nx()))

dfl.to_domain('sf')
P_loss_array2 = np.array([])
P_loss_array_total = np.array([])

# P_entrance = dfl_waveguide2.E()

N = 1666
for i in range(N):
    if i == 1:
        P_entrance = dfl_waveguide2.E()
    P_before=dfl_waveguide2.E()
    dfl_waveguide2 = dfl_ap_circ(dfl_waveguide2, r=r)
    P_after=dfl_waveguide2.E()
    
    if i != 1:
        P_loss_array2 = np.append(P_loss_array2, round((1-((P_after)/P_before))*100, 6))
        P_loss_array_total = np.append(P_loss_array_total, round((1-((P_after)/P_entrance))*100, 6))
    
    dfl_waveguide2.prop(z=z)

I_x_array2 = dfl_waveguide2.intensity()[0,dfl_waveguide2.Ny()//2,:]

plot_dfl(dfl_waveguide2, fig_name='dfl after prop', domains = "fs")
# plot_dfl(dfl_waveguide, fig_name='dfl after prop', domains = "fk")
#%%
omega = 2*np.pi*speed_of_light/lambds
Z = np.arange(1, n) * z
L_p = (1 - np.exp(-4.75 * speed_of_light**(3/2) * z**(1/2) * omega**(-3/2) * r**(-3) * Z))*100

fig, ax1 = plt.subplots()
ax1.plot(range(2, n), P_loss_array2[1:], '--', color='black')   
ax1.set_ylabel('Power losses per cell, %')

ax2 = ax1.twinx()
ax2.plot(range(1, n), P_loss_array_total, label='OCELOT simulation')
ax2.plot(range(1, n), L_p, label='theory')

ax2.set_ylabel('Power losses, %')
ax1.set_xlabel('iris number') 
plt.xlim(0)
plt.ylim(0)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%
Z = np.linspace(0, L, n)
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig, axs = plt.subplots(3, 2, figsize=(16,8), gridspec_kw={'width_ratios': [4, 1]})

axs[0, 0].pcolormesh(Z, dfl_waveguide.scale_y()*1e3, np.log(I_x_array[:,:]))
axs[1, 0].pcolormesh(Z, dfl_waveguide.scale_y()*1e3, I_x_array[:,:])

axs[0, 1].semilogx(I_x_array[:,-1], dfl_waveguide.scale_y()*1e3)
axs[1, 1].plot(I_x_array[:,-1], dfl_waveguide.scale_y()*1e3)

# axs[0, 1].semilogx(I_x_array2, dfl_waveguide.scale_y()*1e3)
# axs[1, 1].plot(I_x_array2, dfl_waveguide.scale_y()*1e3)

axs[0, 0].set_xlabel('Power losses, %')
axs[0, 0].set_ylabel('x, [mm]')
axs[1, 0].set_xlabel('iris line length,[m]')
axs[1, 0].set_ylabel('x, [mm]')

# axs[0, 0].set_xticks(z*np.arange(N))#set_visible(False)
# axs[0, 0].axes.set_xticklabels(loss_array)#set_visible(False)

# axs[0, 1].yaxis.set_visible(False)
# axs[1, 1].yaxis.set_visible(False)

# axs[0, 1].xaxis.tick_top()
axs[0, 1].set_xlabel('log(I), arb.units')
axs[1, 1].set_xlabel('I, arb.units')

# axs[0, 1].set_xlim(0)
axs[1, 1].set_xlim(0)
axs[1, 1].grid()
axs[0, 1].grid()

axs[2, 0].plot(np.arange(2, N)*z, P_loss_array2[1:], '--', color='black')   
axs[2, 0].set_ylabel('Power losses per cell, %')

ax2 = axs[2, 0].twinx()
ax2.plot(np.arange(2, N)*z, P_loss_array_total[1:], label='OCELOT simulation')
ax2.plot(np.arange(2, N)*z, L_p[1:], label='theory')
ax2.legend()
ax2.set_ylabel('Power losses, %')
axs[2, 0].set_xlabel('iris number') 
axs[2, 0].grid()
axs[2, 0].set_xlim(0, N*z)
axs[2, 0].set_ylim(0)
ax2.set_ylim(0)
# plt.xlim(0)
# plt.ylim(0)
fig.delaxes(axs[2, 1])
plt.show() 

#%%   
plot_dfl(dfl_waveguide, fig_name='dfl after prop', domains = "fs")
# plot_dfl(dfl_waveguide, fig_name='dfl after prop', domains = "fk")










