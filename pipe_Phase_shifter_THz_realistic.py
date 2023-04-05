#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:36:34 2022

@author: trebushi
"""

import numpy as np
import matplotlib as plt
from scipy.integrate import quad
from scipy import special
import itertools
from operator import add
import time

from ocelot.gui import *
from ocelot.common.globals import *  # import of constants like "h_eV_s" and
from ocelot.cpbd.elements import *
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.beam import *
from ocelot.rad.screen import Screen
from ocelot.rad.radiation_py import calculate_radiation, track4rad_beam, traj2motion, x2xgaus
from ocelot.optics.wave import dfl_waistscan, screen2dfl, RadiationField
from ocelot.gui.dfl_plot import plot_dfl, plot_dfl_waistscan

from SRinWguide import *

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    # return array[idx], idx

def integrad_wind_func(z, By, gamma, omega, d = 0.05, Fl=6):

    lambdas = 2*np.pi*speed_of_light/omega
    u = np.sqrt((speed_of_light*beta)**2)
    R = (m_e_kg/e_) * gamma * u/np.mean(By)
    Lf = (lambdas * R**2/2/np.pi)**(1/3)
    
    z_gaus = x2xgaus(z)
    
    point = [find_nearest(z_gaus, value=Lf[i]*Fl) for i in range(np.shape(Lf)[0])]
    
    point = np.array(point)
    
    step = [np.full((point[i]), 1) for i in range(np.shape(Lf)[0])] 
    
    n = [np.arange(0, np.shape(z_gaus)[0] - point[i], 1) for i in range(np.shape(Lf)[0])] 
    
    decay_exp = [np.exp(-n[i]*d) for i in range(np.shape(Lf)[0])] 
    
    window_func = np.array([np.concatenate((step[i], decay_exp[i]), axis=0) for i in range(np.shape(Lf)[0])]).T

    return window_func

beam = Beam()
beam.E = 4.0            # beam energy in [GeV]
beam.I = 0.4 
gamma = beam.E/m_e_GeV #0.5109906e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)

e_ = 1.60218e-19 #elementary charge
m_e_ = 9.1093837e-31 # mass of electron in kg

B0 = 2
lperiod = 0.9# [m] undulator period 
nperiods = 1.5
L_w = lperiod * nperiods

K = 0.9336 * B0 * lperiod * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 


# und = Undulator(Kx=K, nperiods=nperiods, lperiod=lperiod, eid="und", phase=-np.pi, end_poles=1)
und = Undulator(Kx=K, nperiods=nperiods, lperiod=lperiod, eid="und", phase=0, end_poles='1/2')

# drift_b = Undulator(Kx=K*0.0000001, nperiods=0.1, lperiod=0.5, eid="drift_b", phase=-np.pi/2, end_poles=0)
# drift_a = Undulator(Kx=K*0.0000001, nperiods=0.1, lperiod=0.5, eid="drift_a", phase=-np.pi/2, end_poles=0)

lambds = (lperiod/2/gamma**2)*(1 + K**2/2)#1e-9

print('E_ph = ', round(speed_of_light * h_eV_s / lambds, 4), 'eV')
# omega = 2 * np.pi * speed_of_light / lambds
THz =  speed_of_light * 1e-12 / lambds#0.000299792458/lambds
print('omega = ', round(THz,1), 'THZ')

if beam.__class__ is Beam:
    p = Particle(x=beam.x, y=beam.y, px=beam.xp, py=beam.yp, E=beam.E)
    p_array = ParticleArray()
    p_array.list2array([p])

lat = MagneticLattice(und)
lat_green = MagneticLattice((und))
# lat_green = MagneticLattice((drift_b, und, drift_a))


tau0 = np.copy(p_array.tau())
p_array.tau()[:] = 0
U, E = track4rad_beam(p_array, lat_green, energy_loss=False, quantum_diff=False, accuracy=0.2)#

# traj = U
# motion = traj2motion(traj)
 
U = np.concatenate(U)
U = np.concatenate(U)
Bx = U[6::9]
By = U[7::9]
x = U[0::9]
xp = U[1::9]
y = U[2::9]
yp = U[3::9]
z  = U[4::9]
# z = x2xgaus(z)

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# fig.suptitle('Magnetic field an trajectory', fontsize=24)
# fig_size = 12
# fig.set_size_inches((2*fig_size,fig_size))

# ax1.plot(z, By, linewidth=3)
# ax1.plot(z, Bx, linewidth=3)

# # ax1.set_ylim(B0-0.1, B0+0.1)
# # plt.plot(z, Bx, linewidth=1)
# # plt.plot(z, Bz, linewidth=1)
# ax1.set_ylabel(r'$B, [T]$', fontsize=18, labelpad = 0.0)
# ax1.grid()

# ax2.plot(z, x*1e6, linewidth=3, color='red')
# # ax2.plot(z, y*1e6, linewidth=3, color='blue')

# ax2.set_ylabel(r'$x, [um]$', fontsize=18, labelpad = 0.0)
# ax2.grid()

# ax3.plot(z, 1e6*xp, linewidth=3, color='orange')
# # ax3.plot(z, 1e6*yp, linewidth=3, color='red')

# ax3.set_ylabel(r'$xp, [urad]$', fontsize=18, labelpad = 0.0)
# ax3.set_xlabel(r'$z, [m]$', fontsize=18, labelpad = 0.0)
# ax3.grid()
# plt.show()

z_hat = 2
z0 = z_hat * L_w
# z0 = 25

theta_hat = 0
xy=theta_hat*z_hat*(np.sqrt(L_w * lambds/2/np.pi))

screen = Screen()
screen.z = z0     # distance from the begining of lattice to the screen 
screen.start_energy = THz*(1 - 0.9)*1e12*h_eV_s #0.000001#1e12*h_eV_s    # [eV], starting photon energy
screen.end_energy =   THz*(1 + 3)*1e12*h_eV_s #0.12#10e12*h_eV_s      # [eV], ending photon energy
screen.num_energy = 501
screen.x = -xy # half of screen size in [m] in horizontal plane
screen.y = -xy   # half of screen size in [m] in vertical plane
screen.nx = 1       # number of poinfts in horizontal plane 
screen.ny = 1    # number of points in vertical plane 

screen = calculate_radiation(lat, screen, beam, energy_loss=False, quantum_diff=False, accuracy=1)
data = screen.Total#/max(screen.Total)
X = screen.Eph

ux = xp*beta*speed_of_light
uy = yp*beta*speed_of_light
uz = np.sqrt((speed_of_light*beta)**2 - ux**2 - uy**2)

omega_r = 2 * np.pi * speed_of_light / lambds
omega = np.linspace(omega_r*(1-0.9), omega_r*(1+3), 251)

# x0 = np.array([0]) #np.linspace(-xy, xy, 1)
# y0 = np.array([0]) #np.linspace(-xy, xy, 1)
x0 = np.linspace(-xy, xy, 1)
y0 = np.linspace(-xy, xy, 1)

m_list = [1]
k_0 = 1
k_list=np.arange(k_0, k_0 + 100, 1)

pipe={'R' : 0.055, 'm_list' : m_list, 'k_list' : k_list}
f_5 = Green_func_integrand(x0, y0, z0, x, y, z, omega, ux, uy, uz, order=3, gradient_term=1, Green_func_type='pipe', **pipe)

# Green = G_free_space(x0, y0, z0, x, y, z, omega)
# nub_G_x, nub_G_y = nubla_G(Green, x0, y0, z0, x, y, z, omega, Green_func_type='free_space')

fx = f_5[1]
# fx = fx * window_func[:, np.newaxis, np.newaxis, :]
plt.figure()
plt.plot((fx[:, 0, 0, -1]))
# plt.plot(np.real(f_5[:, 0, 0, -1]))
plt.show()


E_x, E_y = Green_func_integrator(f_5, z, order=3)

Ix, Iy = SR_field2Intensity(E_x, E_y, I_ebeam=beam.I)
Ix = Ix[0, 0, :] #+ Iy[0,0,:]

f_5_free = Green_func_integrand(x0, y0, z0, x, y, z, omega, ux, uy, uz, order=3, gradient_term=True, Green_func_type='free_space')
E_x_free, E_y_free = Green_func_integrator(f_5_free, z, order=3)
Ix_free, Iy_free = SR_field2Intensity(E_x_free, E_y_free, I_ebeam=beam.I)
Ix_free = Ix_free[0, 0, :] #+ Iy[0,0,:]


def E_ph2THz(x):
    return x / h_eV_s * 1e-12

def THz2E_ph(x):
    return x * h_eV_s / 1e-12

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_title(r"$\hatz$ = " + str(round(z_hat, 2)) + r", $\hat{\theta}$ = " + str(theta_hat))#.format(z_hat, r_hat))

# ax1.plot(X/h_eV_s*1e-12, data, linewidth=1, color='black', linestyle='-', label='OCELOT')

ax1.plot(omega*1e-12/2/np.pi, Ix_free, linewidth=1, color='black', linestyle='-', label='free space')
ax1.plot(omega*1e-12/2/np.pi, Ix, label=r'iris', linestyle='-.', linewidth=1, color='blue', alpha=1)

secax = ax1.secondary_xaxis('top', functions=(THz2E_ph, E_ph2THz))
secax.set_xlabel(r'$E_{ph}$, [eV]', fontsize=14)

ax1.set_yscale('log')
# ax1.set_xscale('log')

ax2 = ax1.twinx() 
# ax2.plot(X/h_eV_s*1e-12, data, linewidth=1, color='black', linestyle='-', label='OCELOT')

ax2.plot(omega*1e-12/2/np.pi, Ix_free, linewidth=1, color='black', linestyle='-', label='free space')
ax2.plot(omega*1e-12/2/np.pi, Ix, label=r'iris', linestyle='-.', linewidth=1, color='blue', alpha=1)
ax2.set_yticks([])

ax3 = ax1.twinx() 
# ax3.plot(omega*1e-12/2/np.pi, np.where(abs((data - Ix)/(data))*1e2<=2, (data - Ix)/(data)*1e2, None), label=r'Norm. discrepancy', linewidth=1, color='red', alpha=1)
# ax3.plot(omega*1e-12/2/np.pi, (data - I)/(data)*1e2, label=r'Norm. discrepancy', linewidth=0.5, color='red', alpha=1)

# ax3.set_yscale('log')
# ax3.set_ylabel('($I_{Green}$ - $I_{oclot}$)/$I_{Green}$, $\%$', fontsize=14, color='red')
# ax3.tick_params(axis='y', colors='red') 
secax3 = ax2.secondary_yaxis(-0.13)
secax3.set_ylabel('Flux, arb.units', fontsize=14)

# ax1.set_ylim(1e4)
# ax1.set_xlim(0, np.max(omega*1e-12/2/np.pi))
ax2.set_ylim(0)
# ax2.set_xlim(0, np.max(omega*1e-12/2/np.pi))

ax1.legend(loc=4)
ax3.legend(loc=4)

ax1.set_xlabel(r'$\Omega$, [THz]', fontsize=14)
# ax1.set_xlabel('E_ph, eV', fontsize=14)
ax1.grid()
plt.tight_layout()
plt.show()
#%%

# THz = 24.5
THz = 2.1# speed_of_light * 1e-12 / lambds#0.000299792458/lambds

# THz = 67
omega = np.array([THz*1e12 * 2 * np.pi])

print(E_ph2THz(omega))

z_hat = 2
z0 = z_hat * L_w

theta_hat = 10
xy = theta_hat*z_hat*(np.sqrt(L_w * lambds/2/np.pi))

n = 151
x0 = np.linspace(-xy, xy, n)
y0 = np.linspace(-xy, xy, n)

m_list = [1]
k_0 = 1
k_list=np.arange(k_0, k_0 + 20, 1)

pipe={'R' : 0.055, 'm_list' : m_list, 'k_list' : k_list}

f = Green_func_integrand(x0, y0, z0, x, y, z, omega, ux, uy, uz, order=3, gradient_term=True, Green_func_type='pipe', **pipe)
E_x, E_y = Green_func_integrator(f, z, order=3) 
Ix, Iy = SR_field2Intensity(E_x, E_y, out="Components", I_ebeam=beam.I)#[0,0,:]
Ix = Ix

f_5_free = Green_func_integrand(x0, y0, z0, x, y, z, omega, ux, uy, uz, order=3, gradient_term=True, Green_func_type='free_space')
E_x_free, E_y_free = Green_func_integrator(f_5_free, z, order=3)
Ix_free, Iy_free = SR_field2Intensity(E_x_free, E_y_free, I_ebeam=beam.I)
I_oclot= Ix_free

#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, num='2D', figsize=(15, 5))
# fig.suptitle("{} THz, {} eV".format(round(THz), round(THz*1e12*h_eV_s, 2)) + r'$, \hat{z}$ = ' + str(z_hat), fontsize=16)
fig.suptitle("{} THz, {} eV".format(round(THz, 2), round(THz*1e12*h_eV_s, 2)) + r'$, z_0$ = ' + str(round(z0, 2)) + ' m', fontsize=16)

theta_x = x0 / (z_hat*(np.sqrt(L_w * lambds/2/np.pi)))
theta_y = y0 / (z_hat*(np.sqrt(L_w * lambds/2/np.pi)))

ax1.pcolormesh(theta_x, theta_y,  I_oclot[:, :, 0])
ax1.set_title("free space")
ax1.set_aspect('equal') 

ax2.pcolormesh(theta_x, theta_y, Ix[:, :, 0])
ax2.set_title("iris")
ax2.set_yticks([])
ax2.set_aspect('equal') 

im = ax3.pcolormesh(theta_x, theta_y, (Ix[:, :, 0] - I_oclot[0, :, :])/np.max(Ix) *1e2, norm=colors.CenteredNorm(), cmap='seismic')
ax3.set_title("Discrepancy")
cax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
fig.colorbar(im, cax=cax, use_gridspec=0)
cax.set_ylabel('($I_{iris}$ - $I_{free space}$)/max($I_{iris}$), $\%$', fontsize=16)
ax3.set_yticks([])
ax3.set_aspect('equal') 

ax1.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
ax2.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
ax1.set_ylabel(r'$\hat{\theta}_y$', fontsize=14)
ax3.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
# plt.tight_layout()
# ax2.set_ylabel('y, [m]')
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, num='1D', figsize=(10, 5),)
fig.suptitle("Slices", fontsize=16)

ax1.plot(theta_x, I_oclot[n//2, :, 0], c='black', label="free space")
ax1.plot(theta_x, Ix[n//2, :, 0], '-.', c='blue', label="iris")
ax1.set_ylim(0)

ax1_twin = ax1.twinx() 
# ax1_twin.plot(theta_x, (Ix[n//2, :, 0] - I_oclot[0, n//2, :])/np.max(Ix) *1e2,  linewidth=0.75, color='red', alpha=1)
# ax1_twin.set_ylabel('Discrepancy, $\%$', fontsize=16, c='red')
ax1_twin.tick_params(axis='y', colors='red') 

# ax1_twin.set_yscale('log')

ax2.plot(theta_y, I_oclot[:, n//2, 0], c='black', label="free space")
ax2.plot(theta_y, Ix[:, n//2, 0], '-.', c='blue', label="iris")
ax2.set_ylim(0)

ax2_twin = ax2.twinx() 
# ax2_twin.plot(theta_y, (Ix[:, n//2, 0] - I_oclot[:, n//2, 0])/np.max(Ix) *1e2, label=r'Norm. discrepancy', linewidth=0.75, color='red', alpha=1)
# ax2_twin.set_ylabel('Discrepancy, $\%$', fontsize=16, c='red')
ax2_twin.tick_params(axis='y', colors='red') 

# ax2_twin.set_yscale('log')

ax1.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
ax2.set_xlabel(r'$\hat{\theta}_y$', fontsize=14)
ax1.set_ylabel('Flux', fontsize=14)
ax2.set_ylabel('Flux', fontsize=14)


ax1.legend()
ax2_twin.legend()
plt.tight_layout()
plt.show()

