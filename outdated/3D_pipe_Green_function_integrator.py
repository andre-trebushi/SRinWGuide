#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:19:15 2022

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

# lambds = 1e-9
beam = Beam()
beam.E = 8.0            # beam energy in [GeV]
beam.I = 0.4 
# speed_of_light = 299792458 #[m/s]
h_bar = 6.582119514e-16 #[eV*s]
gamma = beam.E/0.5109906e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)
e_ = 1.60218e-19 #elementary charge

B0 = 1.18
lperiod = 1.9 # [m] undulator period 
K = 0.9336 * B0 * lperiod * 100
# longitudinal coordinates from 0 to lperiod*nperiods in [mm] 

und = Undulator(Kx=K, nperiods=10, lperiod=lperiod, eid="und", phase=0, end_poles='3/4')

# THz = 3
# lambds = 0.000299792458 / THz

lambds = (lperiod/2/gamma**2)*(1 + K**2/2)#1e-9
# omega = 2 * np.pi * speed_of_light / lambds
THz =  speed_of_light * 1e-12 / lambds#0.000299792458/lambds
print(THz)

if beam.__class__ is Beam:
    p = Particle(x=beam.x, y=beam.y, px=beam.xp, py=beam.yp, E=beam.E)
    p_array = ParticleArray()
    p_array.list2array([p])

lat = MagneticLattice(und)
tau0 = np.copy(p_array.tau())
p_array.tau()[:] = 0
U, E = track4rad_beam(p_array, lat, energy_loss=False, quantum_diff=False, accuracy=0.01)#, end_poles='3/4')

U = np.concatenate(U)
U = np.concatenate(U)
Bx = U[6::9]
By = U[7::9]
x = U[0::9]
xp = U[1::9]
y = U[2::9]
yp = U[3::9]
z  = U[4::9]
#%%
screen = Screen()
screen.z = 30     # distance from the begining of lattice to the screen 

screen.start_energy = 0.000001#0.5e12*h_eV_s/2/np.pi     # [eV], starting photon energy
screen.end_energy =   0.1#25e12*h_eV_s/2/np.pi      # [eV], ending photon energy

screen.num_energy = 1000
screen = calculate_radiation(lat, screen, beam, energy_loss=True, quantum_diff=True, accuracy=0.1)
    
data = screen.Total#/max(screen.Total)
# total = max(screen.Total)
X = screen.Eph

fig, ax1 = plt.subplots()
ax1.plot(X*1e-12/h_eV_s, data, linewidth=0.5, color='black', linestyle='-', label='OCELOT')
# ax1.set_yscale('log')
# ax1.set_xscale('log')

ax1.set_ylim(0)
ax1.legend(loc=0)
ax1.set_xlabel('THz', fontsize=14)
plt.grid()
plt.tight_layout()
plt.show()
#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
fig.suptitle('Magnetic field an trajectory')
fig_size = 12
fig.set_size_inches((2*fig_size,fig_size))

ax1.plot(z, By, linewidth=3)
# plt.plot(z, Bx, linewidth=1)
# plt.plot(z, Bz, linewidth=1)
ax1.set_ylabel(r'$B, [T]$', fontsize=18, labelpad = 0.0)
ax1.grid()

ax2.plot(z, x*1e6, linewidth=3, color='red')
ax2.set_ylabel(r'$x, [um]$', fontsize=18, labelpad = 0.0)
ax2.grid()

ax3.plot(z, 1e6*xp, linewidth=3, color='orange')
ax3.set_ylabel(r'$xp, [urad]$', fontsize=18, labelpad = 0.0)
ax3.grid()
plt.show()

#%%
def delta(omega, a, b):
    # a = 0.3
    # b = 0.5
    beta0 = 0.824 
    return (1 + 1j)*beta0*np.sqrt(speed_of_light*b/4/omega/a**2)

def G(x0, y0, z0, x, y, z, omega, m_list=[0], k_list=[1]):
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
    
    phi0 = np.arctan2(y0, x0)
    phi = np.arctan2(y, x)
    
    r0 = np.sqrt(x0**2 + y0**2)
    r =  np.sqrt(x**2 + y**2)
    
    R = 0.02
    
    # m_list = [0]#range(0, 5)
    # k_list = [3]#range(2, 3)
    
    Green = np.zeros((np.shape(z)[0], np.shape(x0)[0], np.shape(y0)[0], np.shape(omega)[0]))
    for mk in itertools.product(m_list, k_list):
        m = mk[0]
        k = mk[1]
        
        mu_mk = special.jnp_zeros(m, k)[-1]
        nu_mk =  special.jn_zeros(m, k)[-1]
        
        print('shape mu, nu = ', mu_mk, nu_mk, np.shape(mu_mk), np.shape(nu_mk))
        
        if m == 0:
            a_m = 1
        elif m >= 1:
            a_m = 2

        A_TE_mk = np.sqrt(a_m/np.pi)/(special.jv(m, mu_mk)*np.sqrt(mu_mk**2 - m**2))
        A_TM_mk = np.sqrt(a_m/np.pi)/nu_mk/special.jv(m-1, nu_mk)
        
        g_TE = (speed_of_light)/(2j*omega[np.newaxis, :]) * A_TE_mk**2 * (mu_mk/2/R)**2 * np.exp(-(1j*speed_of_light*(z0 - z[:, np.newaxis])*mu_mk**2)/(2*omega[np.newaxis, :]*R**2))
        g_TM = (speed_of_light)/(2j*omega[np.newaxis, :]) * A_TM_mk**2 * (nu_mk/2/R)**2 * np.exp(-(1j*speed_of_light*(z0 - z[:, np.newaxis])*nu_mk**2)/(2*omega[np.newaxis, :]*R**2))
        
        m_1_x1_TE =  special.jv(m-1, mu_mk*r0/R)*np.cos((m-1)*phi0) +  special.jv(m+1, mu_mk*r0/R)* np.cos((m+1)*phi0)
        m_1_y1_TE = -special.jv(m-1, mu_mk*r0/R)*np.sin((m-1)*phi0) +  special.jv(m+1, mu_mk*r0/R)* np.sin((m+1)*phi0)
        m_1_x2_TE =  special.jv(m-1, mu_mk*r/R)* np.cos((m-1)*phi)  +  special.jv(m+1, mu_mk*r/R) * np.cos((m+1)*phi)
        m_1_y2_TE = -special.jv(m-1, mu_mk*r/R)* np.sin((m-1)*phi)  +  special.jv(m+1, mu_mk*r/R) * np.sin((m+1)*phi)
        M1_TE = np.array([[m_1_x1_TE[np.newaxis, :, :] * m_1_x2_TE[:, np.newaxis, np.newaxis], m_1_x1_TE[np.newaxis, :, :] * m_1_y2_TE[:, np.newaxis, np.newaxis]], [m_1_y1_TE[np.newaxis, :, :] * m_1_x2_TE[:, np.newaxis, np.newaxis], m_1_y1_TE[np.newaxis, :, :] * m_1_y2_TE[:, np.newaxis, np.newaxis]]])
        
        m_2_x1_TE = -special.jv(m-1, mu_mk*r0/R)*np.sin((m-1)*phi0) - special.jv(m+1, mu_mk*r0/R)* np.sin((m+1)*phi0)
        m_2_y1_TE = -special.jv(m-1, mu_mk*r0/R)*np.cos((m-1)*phi0) + special.jv(m+1, mu_mk*r0/R)* np.cos((m+1)*phi0)
        m_2_x2_TE = -special.jv(m-1, mu_mk*r/R)* np.sin((m-1)*phi)  - special.jv(m+1, mu_mk*r/R) * np.sin((m+1)*phi)
        m_2_y2_TE = -special.jv(m-1, mu_mk*r/R)* np.cos((m-1)*phi)  + special.jv(m+1, mu_mk*r/R) * np.cos((m+1)*phi)
        M2_TE = np.array([[m_2_x1_TE[np.newaxis, :, :] * m_2_x2_TE[:, np.newaxis, np.newaxis], m_2_x1_TE[np.newaxis, :, :] * m_2_y2_TE[:, np.newaxis, np.newaxis]], [m_2_y1_TE[np.newaxis, :, :] * m_2_x2_TE[:, np.newaxis, np.newaxis], m_2_y1_TE[np.newaxis, :, :] * m_2_y2_TE[:, np.newaxis, np.newaxis]]])

        m_1_x1_TM = special.jv(m-1, nu_mk*r0/R)*np.sin((m-1)*phi0) - special.jv(m+1, nu_mk*r0/R)*np.sin((m+1)*phi0)
        m_1_y1_TM = special.jv(m-1, nu_mk*r0/R)*np.cos((m-1)*phi0) + special.jv(m+1, nu_mk*r0/R)*np.cos((m+1)*phi0)
        m_1_x2_TM = special.jv(m-1, nu_mk*r/R)* np.sin((m-1)*phi)  - special.jv(m+1, nu_mk*r/R) *np.sin((m+1)*phi)
        m_1_y2_TM = special.jv(m-1, nu_mk*r/R)* np.cos((m-1)*phi)  + special.jv(m+1, nu_mk*r/R) *np.cos((m+1)*phi)
        M1_TM = np.array([[m_1_x1_TM[np.newaxis, :, :] * m_1_x2_TM[:, np.newaxis, np.newaxis], m_1_x1_TM[np.newaxis, :, :] * m_1_y2_TM[:, np.newaxis, np.newaxis]], [m_1_y1_TM[np.newaxis, :, :] * m_1_x2_TM[:, np.newaxis, np.newaxis], m_1_y1_TM[np.newaxis, :, :] * m_1_y2_TM[:, np.newaxis, np.newaxis]]])

        m_2_x1_TM =  special.jv(m-1, nu_mk*r0/R)*np.cos((m-1)*phi0) -  special.jv(m+1, nu_mk*r0/R)*np.cos((m+1)*phi0)
        m_2_y1_TM = -special.jv(m-1, nu_mk*r0/R)*np.sin((m-1)*phi0) -  special.jv(m+1, nu_mk*r0/R)*np.sin((m+1)*phi0)
        m_2_x2_TM =  special.jv(m-1, nu_mk*r/R)* np.cos((m-1)*phi)  -  special.jv(m+1, nu_mk*r/R) *np.cos((m+1)*phi)
        m_2_y2_TM = -special.jv(m-1, nu_mk*r/R)* np.sin((m-1)*phi)  -  special.jv(m+1, nu_mk*r/R) *np.sin((m+1)*phi)
        M2_TM = np.array([[m_2_x1_TM[np.newaxis, :, :] * m_2_x2_TM[:, np.newaxis, np.newaxis], m_2_x1_TM[np.newaxis, :, :] * m_2_y2_TM[:, np.newaxis, np.newaxis]], [m_2_y1_TM[np.newaxis, :, :] * m_2_x2_TM[:, np.newaxis, np.newaxis], m_2_y1_TM[np.newaxis, :, :] * m_2_y2_TM[:, np.newaxis, np.newaxis]]])
        
        print(np.shape(M1_TE), np.shape(M2_TE), np.shape(M1_TM), np.shape(M2_TM))
        print(np.shape(g_TE), np.shape(g_TM))
        G_nk = g_TE[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :] * (M1_TE[:, :, :, :, :,np.newaxis] + M2_TE[:, :, :, :, :,np.newaxis]) + g_TM[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :] * (M1_TM[:, :, :, :, :,np.newaxis] + M2_TM[:, :, :, :, :,np.newaxis])
        # G_12 = g_TE[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TE[1,0, :, :, :,np.newaxis] + M2_TE[1,0, :, :, :,np.newaxis]) + g_TM[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TM[1,0, :, :, :,np.newaxis] + M2_TM[1,0, :, :, :,np.newaxis])
        # G_22 = g_TE[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TE[1,1, :, :, :,np.newaxis] + M2_TE[1,1, :, :, :,np.newaxis]) + g_TM[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TM[1,1, :, :, :,np.newaxis] + M2_TM[1,1, :, :, :,np.newaxis])
        # G_21 = g_TE[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TE[0,1, :, :, :,np.newaxis] + M2_TE[0,1, :, :, :,np.newaxis]) + g_TM[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TM[0,1, :, :, :,np.newaxis] + M2_TM[0,1, :, :, :,np.newaxis])
        
        # Green_11 = G_11 + Green_11
        # Green_12 = G_12 + Green_12
        # Green_22 = G_22 + Green_22
        # Green_21 = G_21 + Green_21
        # Green = np.add(G_nk, Green)  #still summing problem here! or not!?  
        Green = G_nk + Green

        print('Green function shape = ', np.shape(Green))
        
    return Green

def nubla_G(Green, x0, y0, z0, x, y, z, h='machine', m_list=[0], k_list=[1]):
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
        bool_x = np.where(x!=0, True, False)
        dx = np.where(x!=0, np.sqrt(epsilon)*abs(x), np.sqrt(epsilon)*1)
        dy = np.where(y!=0, np.sqrt(epsilon)*abs(y), np.sqrt(epsilon)*1)
    elif isinstance(dx, float):
        dx = h
        dy = h
    else: 
        ValueError('please, enter a float "h" for taking derivative')
    dG_x = G(x0, y0, z0, x + dx, y, z, omega, m_list=m_list, k_list=k_list) - Green
    dG_y = G(x0, y0, z0, x, y + dy, z, omega, m_list=m_list, k_list=k_list) - Green
    return dG_x/dx[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis], dG_y/dy[np.newaxis, np.newaxis,:, np.newaxis, np.newaxis, np.newaxis]

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
def gamma_z_integral(u_z, z, omega, order=2):
    I_array = np.array([])
    z_dz = np.roll(z, -1)[:-1]
    dz = np.array(z_dz - z[:-1])
    
    func = f(u_z, omega)
    
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

# B0 = 1
# lperiod = 1# [m] undulator period 
B0 = 0.4
lperiod = 0.05# [m] undulator period 

K = 0.9336 * B0 * lperiod * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 
und = Undulator(Kx=K, nperiods=15, lperiod=lperiod, eid="und", phase=0, end_poles='3/4')

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
U, E = track4rad_beam(p_array, lat, energy_loss=False, quantum_diff=False, accuracy=0.5)#
U = np.concatenate(U)
U = np.concatenate(U)
Bx = U[6::9]
By = U[7::9]
x = U[0::9]
xp = U[1::9]
y = U[2::9]
yp = U[3::9]
z  = U[4::9]

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# fig.suptitle('Magnetic field an trajectory', fontsize=24)
# fig_size = 12
# fig.set_size_inches((2*fig_size,fig_size))

# ax1.plot(z, By, linewidth=3)
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
#%%

ux = xp[:-1]*beta*speed_of_light
uy = yp[:-1]*beta*speed_of_light
u_z = np.sqrt((speed_of_light*beta)**2 - ux**2 - uy**2)

omega_r = 2 * np.pi * speed_of_light / lambds
# omega = np.array([omega_r])
omega = np.linspace(omega_r*(1-0.9), omega_r*(1+1.05), 1000)#np.linspace(2*np.pi*(E-200)/h_eV_s , 2*np.pi*(E+6000)/h_eV_s, 2000)

xy = 0#-0.5e-3#10e-4
x0 = np.linspace(-xy, xy, 1)
y0 = np.linspace(-xy, xy, 1)
z0 = 30

h_erg_s = 1.054571817 * 1e-27 
h_J_s = 1.054571817 * 1e-34 

Int_z = np.array(gamma_z_integral(u_z, z, omega, order=2))
 
Green = 0
# for i in range(1, 100):
i=1
m_list=[1]#np.arange(0, 5, 1)
k_list=np.arange(1, 2, 1)
# n_list = [0]
# k_list = [1]
Green = G(x0, y0, z0, x, y, z, omega, m_list=m_list, k_list=k_list)
print(Green)

nub_G_x, nub_G_y = nubla_G(Green, x0, y0, z0, x, y, z, m_list=m_list, k_list=k_list)
nub_G_x, nub_G_y = nub_G_x[:,:, :-1, :, :, :], nub_G_y[:,:,:-1, :, :, :]
Green = Green[:, :, :-1, :, :, :]

print(np.shape(nub_G_x), np.shape(Green), np.shape(Int_z))

z_dz = np.roll(z, -1)[:-1]
dz = np.array(z_dz - z[:-1])

Gf = lambda ux, Green, Int_z, omega: (4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*(ux[:, np.newaxis, np.newaxis, np.newaxis]*Green[0,0,:,:,:,:] + uy[:, np.newaxis, np.newaxis, np.newaxis]*Green[0,1,:,:,:,:])) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
# Gf = lambda ux, Green, Int_z, omega: (4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*ux[:, np.newaxis, np.newaxis, np.newaxis]*Green) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])

fG = Gf(ux, Green, Int_z, omega)
# E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis] * (fG), axis=0) #oder 2
E_x = np.sum(dz[:-1, np.newaxis, np.newaxis, np.newaxis] * (fG[:-1] + np.roll(fG, -1,  axis=0)[:-1])/2, axis=0) #oder 3
# E_x = np.sum((f[:-1] + 4* + np.roll(f, -1,  axis=0)[:-1]) * dz[:, np.newaxis, np.newaxis, np.newaxis] / 6, axis=0) #oder 4 to be done

speed_of_light_SGS = speed_of_light * 1e2
A = 0.4 * speed_of_light / (e_ * h_erg_s * 4 * np.pi**2) / (1e7/speed_of_light**2) * 1e-2#* SI unit 1e-4 is due to cm^2 -> mm^2, (1e7/speed_of_light**2) is the factor due to transformation of (I/eh_bar) * c * E**2, I -> I * 1e-1 c, e -> e * 1e-1 c, h_bar -> h_bar * 1e7, c -> c * 1e2, E -> E * 1e6 / c 

I = A * (np.real(E_x[0, 0, :])**2 + np.imag(E_x[0, 0, :])**2)
   

z0 = 30

screen = Screen()
screen.z = z0     # distance from the begining of lattice to the screen 
screen.start_energy = THz*(1 - 0.9)*1e12*h_eV_s #0.000001#1e12*h_eV_s    # [eV], starting photon energy
screen.end_energy =   THz*(1 + 1.05)*1e12*h_eV_s #0.12#10e12*h_eV_s      # [eV], ending photon energy
screen.num_energy = 500
screen = calculate_radiation(lat, screen, beam, energy_loss=False, quantum_diff=False, accuracy=1)
data = screen.Total#/max(screen.Total)
X = screen.Eph


    
#%%
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(omega/1e12/2/np.pi, I, linewidth=1, color='blue', label='Green func')

# ax2 = ax1.twinx() 
ax1.plot(X*1e-12/h_eV_s, data, linewidth=1, color='black', linestyle='-', label='OCELOT')

ax1.set_yscale('log')
# ax2.set_yscale('log')

ax1.set_ylabel('$I_{Green \; func}$', color='blue', fontsize=16)
ax2.set_ylabel('$I_{OCELOT}$', color='black', fontsize=16)
ax1.tick_params(axis='y', colors='blue')
ax2.tick_params(axis='y', colors='black')

# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax2.axvline(x=E, color='red')
ax1.set_xlabel('THz', color='black', fontsize=16)

ax1.set_ylim(0)
ax2.set_ylim(0)
ax1.set_xlim(0)
ax2.set_xlim(0)
ax1.legend(loc=1)
# ax1.set_xlabel('omega, THz', fontsize=14)
ax2.legend(loc=2)
ax1.grid(axis='x')

plt.show()























#%%
# ux, uy, uz = u(x, y, z)
ux = xp[:-1]*beta*speed_of_light
uy = yp[:-1]*beta*speed_of_light
# omega = 2 * np.pi * speed_of_light / lambds
# E = speed_of_light / lambds * h_eV_s

# omega = np.linspace(0.5*1e12*2*np.pi, 30*1e12*2*np.pi, 500)#np.linspace(2*np.pi*(E-200)/h_eV_s , 2*np.pi*(E+6000)/h_eV_s, 2000)
omega = np.array([THz*1e12*2*np.pi])
xy = 0.3
x0 = np.linspace(-xy, xy, 100)
y0 = np.linspace(-xy, xy, 100)
z0 = 30


Green = G(x0, y0, z0, x, y, z, omega)
# nub_G_x, nub_G_y = nubla_G(Green, x0, y0, z0, x, y, z, omega)
Green = Green[:,:,:-1, :, :, :]
print(np.shape(Green))

Int_z = np.array(gamma_z_integral(x, y, z, omega))

# print(np.shape(nub_G_x), np.shape(Green), np.shape(Int_z))
# print(np.shape(Green), np.shape(Int_z))
# print(np.shape(Green))
# print(Green)
z_dz = np.roll(z, -1)[:-1]
dz = np.array(z_dz - z[:-1])

#%%
# E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * ((1j*omega[np.newaxis, np.newaxis, np.newaxis, :]/speed_of_light**2)*ux[:, np.newaxis, np.newaxis, np.newaxis]*Green + nub_G_x) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])), axis=0)
E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*(ux[:, np.newaxis, np.newaxis, np.newaxis]*Green[0,0,:,:,:,:] + uy[:, np.newaxis, np.newaxis, np.newaxis]*Green[0,1,:,:,:,:])) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])), axis=0)
# E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * (nub_G_x) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])), axis=0)
I = np.real(E_x[0,0,:])**2 + np.imag(E_x[0,0,:])**2

screen = Screen()
screen.z = 30     # distance from the begining of lattice to the screen 

screen.start_energy = 0.000001#1e12*h_eV_s    # [eV], starting photon energy
screen.end_energy =   0.12#10e12*h_eV_s      # [eV], ending photon energy
screen.num_energy = 1000
screen = calculate_radiation(lat, screen, beam, energy_loss=False, quantum_diff=False, accuracy=0.1)
    
data = screen.Total/max(screen.Total)
total = max(screen.Total)
X = screen.Eph


fig, ax1 = plt.subplots()

ax1.plot(omega/1e12/2/np.pi, I, label='Green func', linewidth=0.5, color='blue')

ax2 = ax1.twinx() 
ax2.plot(X*1e-12/h_eV_s, data, linewidth=0.5, color='black', linestyle='-', label='OCELOT')

ax1.set_yscale('log')
ax2.set_yscale('log')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
# ax2.axvline(x=E, color='red')
ax1.set_ylim(0)
ax2.set_ylim(0)
ax1.set_xlim(0)
ax2.set_xlim(0)
ax1.legend(loc=0)
ax1.set_xlabel('omega, THz', fontsize=14)
ax2.legend(loc=3)
plt.grid()
plt.show()
#%%
# E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * ((1j*omega[np.newaxis, np.newaxis, np.newaxis, :]/speed_of_light**2)*ux[:, np.newaxis, np.newaxis, np.newaxis]*Green + nub_G_x) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])), axis=0)
E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*(ux[:, np.newaxis, np.newaxis, np.newaxis]*Green[0,0,:,:,:,:] + uy[:, np.newaxis, np.newaxis, np.newaxis]*Green[0,1,:,:,:,:])) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])), axis=0)
# E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis]*((4*np.pi*e_/speed_of_light) * (nub_G_x) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])), axis=0)

I = np.real(E_x[0,0,:])**2 + np.imag(E_x[0,0,:])**2
# I = I/np.max(I)
# E_y = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis]*(w(4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*uy[:, np.newaxis, np.newaxis, np.newaxis]*Green + nub_G_y) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])), axis=0)
I = np.real(E_x[:,:,0])**2 + np.imag(E_x[:,:,0])**2

plt.figure('XY, x-pol')
plt.pcolormesh(x0, y0, np.real(E_x[:,:,0])**2 + np.imag(E_x[:,:,0])**2, shading='auto')
plt.tight_layout()
plt.show()

plt.figure('X plot')
plt.plot(x0, I[:, np.shape(y0)[0]//2])
plt.plot(y0, I[np.shape(x0)[0]//2, :])

plt.show()
#%%

plt.figure('XY, x-pol')
plt.pcolormesh(x0*1e6, y0*1e6, np.real(E_x[:,:,0])**2 + np.imag(E_x[:,:,0])**2)
plt.tight_layout()
plt.show()

# plt.figure('X')
# plt.plot(x0*1e6, np.real(E_x[:, np.shape(y0)[0]//2])**2 + np.imag(E_x[:, np.shape(y0)[0]//2])**2)
# plt.show()
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











