#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:09:22 2022

@author: trebushi
"""

import numpy as np
import matplotlib as plt
from scipy.integrate import quad

from ocelot.gui import *
from ocelot.common.globals import *  # import of constants like "h_eV_s" and
from ocelot.cpbd.elements import Undulator
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.beam import *
from ocelot.rad.screen import Screen
from ocelot.rad.radiation_py import calculate_radiation, track4rad_beam, traj2motion
from ocelot.optics.wave import dfl_waistscan, screen2dfl, RadiationField
from ocelot.gui.dfl_plot import plot_dfl, plot_dfl_waistscan

# lambds = 1e-9

# speed_of_light = 299792458 #[m/s]
h_bar = 6.582119514e-16 #[eV*s]
gamma = 3./0.51099890221e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)
e_ = 1.60218e-19 #elementary charge

harm = 7
B0 = 1.08    # [T] amplitude of the magnetic field in the undulators
B0_PS = 0.383 #0.442#0.564# [T] amplitude of the magnetic field in the phase shifter

lperiod = 0.0135 # [m] undulator period 
K = 1 #0.9336 * B0 * lperiod * 100
# longitudinal coordinates from 0 to lperiod*nperiods in [mm] 
beam = Beam()
beam.E = 3.0            # beam energy in [GeV]
beam.I = 0.4  
und = Undulator(Kx=K, nperiods=100, lperiod=lperiod, eid="und", phase=0, end_poles='3/4')

lambds = (lperiod/2/gamma**2)*(1 + K**2/2)#1e-9
omega = 2 * np.pi * speed_of_light / lambds

if beam.__class__ is Beam:
    p = Particle(x=beam.x, y=beam.y, px=beam.xp, py=beam.yp, E=beam.E)
    p_array = ParticleArray()
    p_array.list2array([p])

lat = MagneticLattice((und))
tau0 = np.copy(p_array.tau())
p_array.tau()[:] = 0
U, E = track4rad_beam(p_array, lat, energy_loss=True, quantum_diff=True, accuracy=1)#, end_poles='3/4')

U = np.concatenate(U)
U = np.concatenate(U)
Bx = U[6::9]
By = U[7::9]
x = U[0::9]
xp = U[1::9]
y = U[2::9]
yp = U[3::9]
z  = U[4::9]

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
# fig.suptitle('Magnetic field an trajectory')
# fig_size = 12
# fig.set_size_inches((2*fig_size,fig_size))

# ax1.plot(z, By, linewidth=3)
# # plt.plot(z, Bx, linewidth=1)
# # plt.plot(z, Bz, linewidth=1)
# ax1.set_ylabel(r'$B, [T]$', fontsize=18, labelpad = 0.0)
# ax1.grid()

# ax2.plot(z, x*1e6, linewidth=3, color='red')
# ax2.set_ylabel(r'$x, [um]$', fontsize=18, labelpad = 0.0)
# ax2.grid()

# ax3.plot(z, 1e6*xp, linewidth=3, color='orange')
# ax3.set_ylabel(r'$xp, [urad]$', fontsize=18, labelpad = 0.0)
# ax3.grid()
# plt.show()
#%%

def G(x0, y0, z0, x, y, z):
    '''
    Parameters
    ----------
    z0 : float
        the longitudinal position at which the field will be calculated.
    x0, y0 : float
        transverse position at which the field will be calculated.

    z : array
        integration position along the electron trajectory.
    x : array
        transverse position along the electron trajectory.

    Returns
    -------
    array with a shape np.shape(z)[0]
        Green function
    '''
    x0, y0 = np.meshgrid(x0, y0)
    return (-1/(4*np.pi*(z0 - z[:, np.newaxis, np.newaxis]))) * np.exp(1j*omega*((x0[np.newaxis, :, :] - x[:, np.newaxis, np.newaxis])**2 + (y0[np.newaxis, :, :] - y[:, np.newaxis, np.newaxis])**2)/(2 * speed_of_light * (z0 - z[:, np.newaxis, np.newaxis])))


def nubla_G(x0, y0, z0, x, y, z):
    '''
    Parameters
    ----------
    z0 : float
        the longitudinal position at which the field will be calculated.
    x0, y0 : float
        transverse position at which the field will be calculated.

    z : array
        integration position along the electron trajectory.
    x : array
        transverse position along the electron trajectory.

    Returns
    -------
    array with shape (2, np.shape(z)[0])
        gradient components of Green function

    '''
    z_dz = np.roll(z, -1)[:-1]
    x_dx = np.roll(x, -1)[:-1]
    y_dy = np.roll(y, -1)[:-1]
    dG = G(x0, y0, z0, x_dx, y_dy, z_dz) - G(x0, y0, z0, x[:-1], y[:-1], z[:-1])
    
    dx_ov_dz = (x_dx - x[:-1])/(z_dz - z[:-1])
    dy_ov_dz = (y_dy - y[:-1])/(z_dz - z[:-1])
    dG_ov_dz = dG/((z_dz - z[:-1])[:, np.newaxis, np.newaxis])
    
    dz_ov_dx = np.where(dx_ov_dz==0, 0, 1/dx_ov_dz)
    dz_ov_dy = np.where(dy_ov_dz==0, 0, 1/dy_ov_dz)
    return dG_ov_dz*dz_ov_dx[:, np.newaxis, np.newaxis], dG_ov_dz*dz_ov_dy[:, np.newaxis, np.newaxis]

def u(x, y, z):
    '''
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    x_dx = np.roll(x, -1)[:-1]
    y_dy = np.roll(y, -1)[:-1]
    z_dz = np.roll(z, -1)[:-1]
    
    u_x = (x_dx - x[:-1])*speed_of_light*beta/(z_dz - z[:-1])
    u_y = (y_dy - y[:-1])*speed_of_light*beta/(z_dz - z[:-1])
    u_z = np.sqrt((speed_of_light*beta)**2 - u_x**2 - u_y**2)
    return u_x, u_y, u_z 

def gamma_z(x, y, z):
    '''
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    _, _, u_z = u(x, y, z)
    return 1/np.sqrt(1 - u_z**2/speed_of_light**2)

f = lambda x, y, z: np.array((omega/2/speed_of_light)*(1/gamma_z(x, y, z)**2))
def gamma_z_integral(x, y, z):
    I_array = np.array([])
    z_dz = np.roll(z, -1)[:-1]
    dz = np.array(z_dz - z[:-1])
    I_array = [np.sum((f(x, y, z)[1:i] + np.roll(f(x, y, z), -1)[1:i])*dz[1:i]/2) for i in range(np.shape(z)[0])]
        
    return I_array



#%%
ux, uy, uz = u(x, y, z)

xy = 1000
x0 = np.linspace(-xy*1e-6, xy*1e-6, 100)
y0 = np.linspace(-xy*1e-6, xy*1e-6, 100)
z0 = 20

nub_G_x, nub_G_y = nubla_G(x0, y0, z0, x, y, z)
Green = G(x0, y0, z0, x, y, z)[1:, :, :]
Int_z = np.array(gamma_z_integral(x, y, z)[1:])

#%%
z_dz = np.roll(z, -1)[:-1]
dz = np.array(z_dz - z[:-1])
E_x = np.sum(dz[:, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*ux[:, np.newaxis, np.newaxis]*Green + nub_G_x) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis])), axis=0)
E_y = np.sum(dz[:, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*uy[:, np.newaxis, np.newaxis]*Green + nub_G_y) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis])), axis=0)

# E_x = np.sum(dz[:, np.newaxis, np.newaxis]*(nub_G_x) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis]), axis=0)
# E_x = np.sum(dz[:, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*ux[:, np.newaxis, np.newaxis]*Green) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis])), axis=0)

plt.figure('XY, x-pol')
plt.pcolormesh(x0*1e6, y0*1e6, np.real(E_x)**2 + np.imag(E_x)**2)
plt.show()


plt.figure('X')
plt.plot(x0*1e6, np.real(E_x[:, np.shape(y0)[0]//2])**2 + np.imag(E_x[:, np.shape(y0)[0]//2])**2)
plt.show()
#%%
I = np.array([])
I = gamma_z_integral(x, y, z)
I = np.array(I)
plt.figure()
# plt.plot(z, np.real(np.exp(1j*I)))
plt.plot(z, I)

# plt.plot(z[:-1], f(x, y, z))
plt.show()


#%%
x0 = np.linspace(-200e-6, 200e-6, 300)
y0 = np.linspace(-200e-6, 200e-6, 3)
z0 = 10
# x0, y0 = np.meshgrid(x0, y0)
# G =  (-1/(4*np.pi*(z0 - z[:, np.newaxis, np.newaxis]))) * np.exp(1j*omega*((x0[np.newaxis, :, :] - x[:, np.newaxis, np.newaxis])**2 + (y0[np.newaxis, :, :] - y[:, np.newaxis, np.newaxis])**2)/(2 * speed_of_light * (z0 - z[:, np.newaxis, np.newaxis])))
nub_G_x, nub_G_y = nubla_G(x0, y0, z0, x, y, z)


# fig = plt.figure('Green function', figsize=(10, 5))
fig, axis = plt.subplots(1, 2, figsize=(13,5))

ax1 = axis[0]
ax2 = axis[1]

ln1 = ax1.plot(x0*1e6, np.real(nub_G_x[-1, np.shape(y0)[0]//2, :]), color='blue', linestyle = '--', label='real(G`)')
ln2 = ax1.plot(x0*1e6, np.imag(nub_G_x[-1, np.shape(y0)[0]//2, :]), color='red', label='imag(G`)')
plt.legend()

ax1_tw = ax1.twinx()
ln3 = ax1_tw.plot(x0*1e6, np.real(G(x0, y0, z0, x, y, z)[-1,np.shape(y0)[0]//2,:]), color='green', label='real(G)')
ln4 = ax1_tw.plot(x0*1e6, np.imag(G(x0, y0, z0, x, y, z)[-1,np.shape(y0)[0]//2,:]), color='orange', label='imag(G)')

# lns = ln1 + ln2 + ln3 + ln4
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)



ln1 = ax2.plot(y0*1e6, np.real(nub_G_y[-1, :, np.shape(x0)[0]//2]), color='blue', linestyle = '--', label='real(G`)')
ln2 = ax2.plot(y0*1e6, np.imag(nub_G_y[-1, :, np.shape(x0)[0]//2]), color='red', label='imag(G`)')
plt.legend()

ax2_tw = ax2.twinx()
ln3 = ax2_tw.plot(y0*1e6, np.real(G(x0, y0, z0, x, y, z)[-1,:,np.shape(x0)[0]//2]), color='green', label='real(G)')
ln4 = ax2_tw.plot(y0*1e6, np.imag(G(x0, y0, z0, x, y, z)[-1,:,np.shape(x0)[0]//2]), color='orange', label='imag(G)')

# lns = ln1 + ln2 + ln3 + ln4
# labs = [l.get_label() for l in lns]
# ax2.legend(lns, labs, loc=0)
# plt.grid()
plt.tight_layout()
plt.show()











