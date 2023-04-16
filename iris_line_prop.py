#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:30:53 2023

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


from ocelot.gui import *
from ocelot.common.globals import *  # import of constants like "h_eV_s" and
from ocelot.cpbd.elements import Undulator
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.beam import *
from ocelot.rad.screen import Screen
from ocelot.rad.radiation_py import calculate_radiation, track4rad_beam, traj2motion
from ocelot.optics.wave import dfl_waistscan, screen2dfl, RadiationField
from ocelot.gui.dfl_plot import plot_dfl, plot_dfl_waistscan

from tqdm import tqdm
from SRinWGuide import *

import xraylib
import math
import logging
import matplotlib


# ebeam parameters
beam = Beam()
beam.E = 4.0            # beam energy in [GeV]
beam.I = 0.4 
gamma = beam.E/0.51099890221e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)

# undulator parameters
B0 = 2
lperiod = 0.9# [m] undulator period 
nperiods = 3
L_w = lperiod * nperiods
K = 0.9336 * B0 * lperiod * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 

und = Undulator(Kx=K, nperiods=nperiods, lperiod=lperiod, eid="und", end_poles='3/4')

xlamds = (lperiod/2/gamma**2)*(1 + K**2/2)
E_ph = speed_of_light * h_eV_s / xlamds
THz = E_ph2THz(E_ph)
#%%
lat = MagneticLattice(und)

motion = track2motion(beam, lat, energy_loss=0, quantum_diff=0, accuracy=0.1)
plot_motion(motion)

#%%
dfl = generate_SR_Green_dfl(lat, beam, L_w + 3, shape=(301, 301, 1), dgrid_xy=(0.13, 0.13), E_ph=E_ph,
                      gradient_term=0, order=3, Green_func_type='free_space', polarization='x', accuracy=0.1)

    #%%
plot_dfl(dfl,fig_name='dfl at source domain', domains = "fs")
plot_dfl(dfl, fig_name='dfl at source k domain', domains = "fk")

#%%
dfl1 = deepcopy(dfl)

dfl_exit = dfl1.prop_m(0, m=1, fine=1, debug=0, return_result=1)
plot_dfl(dfl_exit,  fig_name='dfl_iris_prop', domains = "fs")
#%%
n=40
z_pos = np.linspace(-L_w, L_w, n)
# z_pos = np.linspace(-1*L_w, -1.5*L_w, n)

m_list = np.linspace(1, 15, n)
# m = 1 + 2**(-100)
I = []#np.array([])
phase = []
max_wig_list = []

dfl2 = deepcopy(dfl_exit)
dfl_prop = deepcopy(dfl_exit)
for z, m in zip(z_pos, m_list):
    dfl_prop = dfl2.prop_m(z, m=1, fine=0, debug=0, return_result=1)
    dfl_prop.to_domain('sf')
    I_shot =0
    # I_shot  = dfl_prop.fld[0, dfl_prop.shape()[1]//2, :]
    I_shot  = dfl_prop.intensity()[0, dfl_prop.shape()[1]//2 + 1, :]
    I.append(np.real(I_shot)/np.max(np.real(I_shot)))
    
    wig = dfl2wig(dfl_prop, domain='x')
    max_wig = np.max(wig.wig)
    max_wig_list.append(max_wig)
    
    phase.append(np.angle(dfl_prop.fld[0, dfl_prop.shape()[1]//2 + 1, :]))
    
    # I.append(dfl_prop.intensity()[0, :, dfl_prop.shape()[2]//2 + 1])

I = np.array(I) 
phase =  np.array(phase)
max_wig_list = np.array(max_wig_list)
#%%
fig, axs = plt.subplots(1, figsize=(12,7.5))
fig.canvas.set_window_title('intensity')
# axs.pcolormesh(z_pos, dfl_2|d_source.scale_x()*1e6, np.sqrt(I.T), cmap='Greys', shading = 'gouraud')
# plt.title(r'$N_w = {}$'.format(nperiods) + r', Aperture = {}'.format(ap) + r'$\cdot (\lambdabar/L_w)^{1/2}$', fontsize=22)

array_lim = np.amax(I)
print(array_lim)
v_max = array_lim
v_min = -array_lim

cmap = plt.cm.get_cmap('seismic')
norm = plt.Normalize(vmin=v_min, vmax=v_max)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

# z_pos = np.linspace(-L_w, L_w*2, n)

im = axs.pcolormesh(z_pos/L_w, dfl_prop.scale_x() / np.sqrt(xlamds * L_w/2/np.pi), I.T/np.max(I), cmap='seismic', shading='auto', vmax=v_max, vmin=v_min)
# im = axs.pcolormesh(z_pos/L_w, dfl_prop.scale_x()*1e6, I.T/np.max(I), cmap='seismic', shading='auto', vmax=v_max, vmin=v_min)

# im = axs.pcolormesh(np.flip(I.T/np.max(I), axis=1), cmap='seismic', shading='auto')
# im = axs.imshow(I.T/np.max(I), 
#             extent=[np.min(z_pos), np.max(z_pos), np.min(dfl_prop.scale_x()*1e6), np.max(dfl_prop.scale_x()*1e6)], 
#             cmap='seismic', aspect='auto')#, shading = 'gouraud')

# cbar = fig.colorbar(sm, label=r'Re(E), arb.unit')
cbar = fig.colorbar(sm, label=r'$|E|^2$, arb.unit')

cbar.ax.tick_params(labelsize=22)  # set font size of color bar labels
# cbar.set_label(r'Re(E), arb.unit', fontsize=22)
cbar.set_label(r'$|E|^2$, arb.unit', fontsize=22)

axs.set_ylabel(r'x/ $(L_w \lambdabar)^{1/2}$', fontsize=18, color='black')
# axs.set_ylabel(r'x, um', fontsize=18, color='black')

axs.set_xlabel('z/$L_w$', fontsize=18, color='black')
# axs[0].set_ylim(4300, 5600)
# axs[0].set_xlim(-2, 2)
# axs.set_ylim(-10, 10)
# axs.set_ylim(-4, 4)

# axs.text(1.5, 4400, 'a)', fontsize=18)#, 
                     # horizontalalignment='left', verticalalignment='top', rotation=270, fontsize=14)  
# plt.tight_layout()

# plt.savefig('/Users/trebushi/Documents/XFEL/project_two_phased_und_MAXIV/pic/intensity_scan_{}'.format(sim_type), dpi=300, bbox_inches='tight')
plt.show() 


#%%

N = 700
z = 0.3
r = 11/2 * 1e-2

L = N*z

n = N*2
dl = L/n

dfl_waveguide = deepcopy(dfl_exit)
j=0
dfl_waveguide.to_domain('sf')

I_x_array=np.zeros((dfl_waveguide.Nx(), n))
I_y_array=np.zeros((dfl_waveguide.Ny(), n))

total_loss_array = np.array([])
rad_left_array = np.array([])
loss_per_cell_array = np.array([])
dfl_waveguide = dfl_ap_circ(dfl_waveguide, r=r)
#%%
for i in range(n):
    if i == 0:
        P_entrance = dfl_waveguide.E()
    
    if (dl*i > j*z):
        P_before=dfl_waveguide.E()
        
        dfl_waveguide = dfl_ap_circ(dfl_waveguide, r=r)
        P_after=dfl_waveguide.E()
        
        total_loss_array = np.append(total_loss_array, round((1-((P_after)/P_entrance))*100, 2))
        
        rad_left_array = np.append(rad_left_array, round(P_after/P_entrance*100, 2))

        loss_per_cell_array = np.append(loss_per_cell_array, round((1-P_after/P_before)*100, 2))

        
        j = j+1
        
    dfl_waveguide.prop_m(z=dl, m=1)
    # dfl_waveguide = dfl_ap_circ(dfl_waveguide, r=0.95*dfl_waveguide.Lx()/2)

    dfl_waveguide.to_domain('sf')
    I_x_array[:, i] = dfl_waveguide.intensity()[0,dfl_waveguide.Ny()//2,:]
    I_y_array[:, i] = dfl_waveguide.intensity()[0,:,dfl_waveguide.Nx()//2]


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

cells = np.arange(0, N)*z
axs[2, 0].plot(cells, loss_per_cell_array, '--', color='black', label='Losses per cell')   
axs[2, 0].set_ylabel('Losses per cell, %')
axs[2, 0].set_xlabel('iris number') 
axs[2, 0].grid()

axs[2, 0].set_xlim(0, N*z)
axs[2, 0].set_ylim(0)

ax2 = axs[2, 0].twinx()
ax2.plot(cells, rad_left_array, '--', color='red', label='Power left')   
ax2.plot(cells, total_loss_array, label='Total losses')
# ax2.plot(np.arange(2, N)*z, L_p[1:], label='theory')
ax2.legend()
ax2.set_ylabel('Power losses, %')

ax2.set_ylim(0)
# plt.xlim(0)
# plt.ylim(0)
fig.delaxes(axs[2, 1])
plt.show() 








