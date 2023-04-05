#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:32:00 2022

@author: trebushi
"""

import numpy as np
import matplotlib as plt
from scipy.integrate import quad
from scipy import special
import itertools
from operator import add

from ocelot.gui import *
from ocelot.common.globals import *  # import of constants like "h_eV_s" and
from ocelot.cpbd.elements import Undulator
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.beam import *
from ocelot.rad.screen import Screen
from ocelot.rad.radiation_py import calculate_radiation, track4rad_beam, traj2motion
from ocelot.optics.wave import dfl_waistscan, screen2dfl, RadiationField
from ocelot.gui.dfl_plot import plot_dfl, plot_dfl_waistscan

def G(x0, y0, z0, x, y, z, omega):
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
    return (-1/(4*np.pi*(z0 - z[:, np.newaxis, np.newaxis, np.newaxis]))) * \
        np.exp(1j*omega[np.newaxis, np.newaxis, np.newaxis, :] * ((x0[np.newaxis, :, :, np.newaxis] - x[:, np.newaxis, np.newaxis, np.newaxis])**2 + (y0[np.newaxis, :, :, np.newaxis] - y[:, np.newaxis, np.newaxis, np.newaxis])**2)/(2 * speed_of_light * (z0 - z[:, np.newaxis, np.newaxis, np.newaxis])))


def nubla_G(Green, x0, y0, z0, x, y, z, h='machine'):
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
    epsilon=1.0
    while epsilon+1>1:
        epsilon=epsilon/2
    epsilon=epsilon*2
    # print("The value of epsilon is: ", epsilon)
    
    if h=='machine':
        dx = np.where(x!=0, np.sqrt(epsilon)*abs(x), np.sqrt(epsilon)*1)
        dy = np.where(y!=0, np.sqrt(epsilon)*abs(y), np.sqrt(epsilon)*1)
    elif isinstance(dx, float):
        dx = h
        dy = h
    else: 
        ValueError('please, enter a float "h" for taking derivative')
    dG_x = G(x0, y0, z0, x + dx, y, z, omega) - Green
    dG_y = G(x0, y0, z0, x, y + dy, z, omega) - Green
    return dG_x/dx[:, np.newaxis, np.newaxis, np.newaxis], dG_y/dy[:, np.newaxis, np.newaxis, np.newaxis]

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
    dz = z_dz - z[:-1]
    u_x = xp*speed_of_light*beta#(x_dx - x[:-1])*speed_of_light*beta/dz
    u_y = yp*speed_of_light*beta#(y_dy - y[:-1])*speed_of_light*beta/dz

    u_z = np.sqrt((speed_of_light*beta)**2 - u_x**2 - u_y**2)
    return u_x, u_y, u_z 

def gamma_z(u_z):
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
    return 1/np.sqrt(1 - (u_z/speed_of_light)**2)

f = lambda u_z, omega: np.array((omega[np.newaxis, :]/2/speed_of_light)*(1/gamma_z(u_z)[:, np.newaxis]**2))
def gamma_z_integral(u_z, z, omega, order=4):
    I_array = np.array([])
    z_dz = np.roll(z, -1)[:-1]
    dz = np.array(z_dz - z[:-1])
    print('omega shape = ', np.shape(omega))
    func = f(u_z, omega)
    print('func shape = ', np.shape(func))

    if order==2:
        I_array = [np.sum(func[0:i]*dz[0:i, np.newaxis], axis=0) for i in range(np.shape(z)[0]-1)]
    elif order==3:
        I_array = [np.sum((func[0:i] + np.roll(func, -1, axis=0)[0:i]) * dz[0:i, np.newaxis]/2, axis=0) for i in range(np.shape(z)[0]-1)]
    elif order==4:
        u_z_dz = np.roll(u_z, -1)[:-1]
        func_dfunc = f((u_z[:-1] + u_z_dz)/2, omega)
        I_array = [np.sum((func[0:i] + 4*func_dfunc[0:i] + np.roll(func, -1, axis=0)[0:i]) * dz[0:i, np.newaxis]/6, axis=0) for i in range(np.shape(z)[0]-1)]
    else:
        print('the integration order might be O(h^2), O(h^3), O(h^4), please, enter 2,3,4 correspondingly')
    return I_array

#%%
# lambds = 1e-9
beam = Beam()
beam.E = 4.0            # beam energy in [GeV]
# beam.E = 8.0            # beam energy in [GeV]
beam.I = 0.4 
gamma = beam.E/0.5109906e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)
e_ = 1.60218e-19 #elementary charge

B0 = 0.4
lperiod = 0.05# [m] undulator period 

K = 0.9336 * B0 * lperiod * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 
und = Undulator(Kx=K, nperiods=50, lperiod=lperiod, eid="und", phase=0)#, end_poles='3/4')

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
tau0 = np.copy(p_array.tau())
p_array.tau()[:] = 0
U, E = track4rad_beam(p_array, lat, energy_loss=False, quantum_diff=False, accuracy=1)#, end_poles='3/4')

U = np.concatenate(U)
U = np.concatenate(U)
Bx = U[6::9]
By = U[7::9]
x = U[0::9]
xp = U[1::9]
y = U[2::9]
yp = U[3::9]
z  = U[4::9]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
fig.suptitle('Magnetic field an trajectory', fontsize=24)
fig_size = 12
fig.set_size_inches((2*fig_size,fig_size))

ax1.plot(z, By, linewidth=3)
# plt.plot(z, Bx, linewidth=1)
# plt.plot(z, Bz, linewidth=1)
ax1.set_ylabel(r'$B, [T]$', fontsize=18, labelpad = 0.0)
ax1.grid()

ax2.plot(z, x*1e6, linewidth=3, color='red')
# ax2.plot(z, y*1e6, linewidth=3, color='blue')

ax2.set_ylabel(r'$x, [um]$', fontsize=18, labelpad = 0.0)
ax2.grid()

ax3.plot(z, 1e6*xp, linewidth=3, color='orange')
# ax3.plot(z, 1e6*yp, linewidth=3, color='red')

ax3.set_ylabel(r'$xp, [urad]$', fontsize=18, labelpad = 0.0)
ax3.set_xlabel(r'$z, [m]$', fontsize=18, labelpad = 0.0)
ax3.grid()
plt.show()

#%%
# ux, uy, uz = u(x, y, z)
ux = xp[:-1]*beta*speed_of_light
uy = yp[:-1]*beta*speed_of_light
u_z = np.sqrt((speed_of_light*beta)**2 - ux**2 - uy**2)

# omega = 2 * np.pi * speed_of_light / lambds
# E = speed_of_light / lambds * h_eV_s

omega_r = 2 * np.pi * speed_of_light / lambds
# omega = np.linspace(0.5*1e12*2*np.pi, 30*1e12*2*np.pi, 500)#np.linspace(2*np.pi*(E-200)/h_eV_s , 2*np.pi*(E+6000)/h_eV_s, 2000)
# omega = np.array([THz*(1 +0.3)*1e12*2*np.pi])
omega = np.array([omega_r])
# omega = np.linspace(omega_r*(1-0.9), omega_r*(1+3), 500)#np.linspace(2*np.pi*(E-200)/h_eV_s , 2*np.pi*(E+6000)/h_eV_s, 2000)

xy = 1.5e-3
x0 = np.linspace(-xy, xy, 200)
y0 = np.linspace(-xy, xy, 100)
z0 = 30

Green = G(x0, y0, z0, x, y, z, omega)
nub_G_x, nub_G_y = nubla_G(Green, x0, y0, z0, x, y, z)

nub_G_x, nub_G_y = nub_G_x[:-1, :, :, :], nub_G_y[:-1, :, :, :]
Green = Green[:-1, :, :, :]

Int_z = np.array(gamma_z_integral(u_z, z, omega, order=4))
print(np.shape(nub_G_x), np.shape(Green), np.shape(Int_z))

z_dz = np.roll(z, -1)[:-1]
dz = np.array(z_dz - z[:-1])
#%%
h_erg_s = 1.054571817 * 1e-27 
h_J_s = 1.054571817 * 1e-34 

E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*ux[:, np.newaxis, np.newaxis, np.newaxis]*Green + nub_G_x) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])), axis=0)
A = 0.4 * speed_of_light / (e_ * h_erg_s * 4 * np.pi**2) / (1e7/speed_of_light**2) * 1e-2#* f*ckn' SI unit 1e-4 is due to cm^2 -> mm^2, (1e7/speed_of_light**2) is the factor due to transformation of (I/eh_bar) * c * E**2, I -> I * 1e-1 c, e -> e * 1e-1 c, h_bar -> h_bar * 1e7, c -> c * 1e2, E -> E * 1e6 / c 
# I = A*(np.real(E_x[:, np.shape(y0)[0]//2, :])**2 + np.imag(E_x[:, np.shape(y0)[0]//2, :])**2)
I = A*(np.real(E_x[:, :, 0])**2 + np.imag(E_x[:, :, 0])**2)

plt.figure('XY, x-pol')
plt.pcolormesh(x0, y0, np.real(E_x[:,:,0])**2 + np.imag(E_x[:,:,0])**2)
plt.tight_layout()
plt.show()

#%%
screen = Screen()
screen.z = z0      # distance from the begining of lattice to the screen 
screen.size_x = xy # half of screen size in [m] in horizontal plane
screen.size_y = xy   # half of screen size in [m] in vertical plane
screen.nx = 200       # number of poinfts in horizontal plane 
screen.ny = 100    # number of points in vertical plane 

omega_r = 2 * np.pi * speed_of_light / lambds
omega_r = omega_r#*(1 - 0.00015)
screen.start_energy = omega_r*h_eV_s/2/np.pi # THz*1e12*h_eV_s# [eV], starting photon energy
screen.end_energy =   omega_r*h_eV_s/2/np.pi #THz*1e12*h_eV_s     # [eV], ending photon energy
screen.num_energy = 1  
screen = calculate_radiation(lat, screen, beam, energy_loss=False, quantum_diff=False, accuracy=1)
# show_flux(screen, unit="mrad",  nfig=13)


#%%
dfl = RadiationField()
dfl = screen2dfl(screen, polarization='x')

current = beam.I
constQuant = 3*alpha/q_e/(4*pi**2)*1e-3 * current * gamma * gamma / z0 / z0 * 1e-6
# dfl.fld = dfl.fld * constQuant

plot_dfl(dfl, domains='s')

fig, ax1 = plt.subplots(figsize=(6, 5))

ax2 = ax1.twinx() 

ax1.plot(screen.Xph*1e-3, constQuant*dfl.intensity()[0, screen.ny//2, :], color='black', label='OCELOT', linewidth=1, linestyle='--')
ax1.plot(x0, I[screen.ny//2, :], color='blue', label='Green func', linewidth=1)
# ax1.set_yscale('log')
ax1.legend(loc=0)

plt.show()








