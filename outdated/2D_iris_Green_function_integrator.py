#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:13:27 2022

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

def G(x0, y0, z0, x, y, z, omega, n_list=[0], k_list=[1]):
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
    
    a = 0.2
    b = 0.15
    
    beta0 = 0.824 
    delta = (1 + 1j) * beta0 * np.sqrt((speed_of_light * b)/(4 * omega * a**2))
    # print(np.shape(delta(omega, a, b)))
    # n_list = [0]#range(-5, 5)
    # k_list = [2]#range(1, 5)
    
    Green = np.zeros((np.shape(z)[0], np.shape(x0)[0], np.shape(y0)[0], np.shape(omega)[0]))
    
    for nk in itertools.product(n_list, k_list):
        
        n = nk[0]
        k = nk[1]
        v_nk = special.jn_zeros(n, k)[-1]
        print('n =', n, 'k =', k, 'v_nk =', v_nk)
        
        G_nk = -((1j * speed_of_light * (1.0 - 2*delta[np.newaxis, np.newaxis, np.newaxis, :])) / (2 * np.pi * omega[np.newaxis, np.newaxis, np.newaxis, :] * a**2)) * \
            (np.exp(-1j*n*(phi0[np.newaxis, :, :, np.newaxis] - phi[:, np.newaxis, np.newaxis, np.newaxis]))) / (special.jv(n+1, v_nk))**2 * \
                np.exp(-(1j*speed_of_light*(z0 - z[:,np.newaxis,np.newaxis,np.newaxis]) * v_nk**2 * (1.0 - 2.0*delta[np.newaxis,np.newaxis,np.newaxis,:])) / (2 * omega[np.newaxis,np.newaxis,np.newaxis,:] * a**2)) * \
                    special.jv(n, v_nk * (1 - delta[np.newaxis,np.newaxis,np.newaxis,:])*r[:, np.newaxis, np.newaxis, np.newaxis]/a) * \
                        special.jv(n, v_nk * (1 - delta[np.newaxis,np.newaxis,np.newaxis,:])*r0[np.newaxis, :, :, np.newaxis]/a)
        
        #print('exp=', np.max(np.real(np.exp(-(1j*speed_of_light*(z0 - z[:,np.newaxis,np.newaxis,np.newaxis]) * v_nk**2 * (1.0 - 2.0*delta[np.newaxis,np.newaxis,np.newaxis,:])) / (2 * omega[np.newaxis,np.newaxis,np.newaxis,:] * a**2)))))

        # G_nk = -((-1j * speed_of_light*(1. - 2 * delta(omega)[np.newaxis, np.newaxis, np.newaxis, :]))/(2 * np.pi * omega[np.newaxis, np.newaxis, np.newaxis, :] * a**2)) * \
        #     (np.exp(-1j*n*(phi0[np.newaxis, :, :, np.newaxis] - phi[:, np.newaxis, np.newaxis, np.newaxis])))/(special.jn(n+1,v_nk))**2

        # G_nk = (np.exp(-1j*(phi0[np.newaxis, :, :, np.newaxis] - phi[:, np.newaxis, np.newaxis, np.newaxis])))
        # G_nk = (np.exp(-1j*(r0[np.newaxis, :, :, np.newaxis] - r[:, np.newaxis, np.newaxis, np.newaxis])/(z0 - z[:, np.newaxis, np.newaxis, np.newaxis])))
        
        # Green = G_nk        
        # print('G_nk function shape = ', np.shape(G_nk))
        
        # Green = G_nk + Green
        Green = np.add(Green, G_nk)  #still summing problem here! or not!?  
        # print('Green function shape = ', np.shape(Green))
        
    return np.array(Green)

def nubla_G(Green, x0, y0, z0, x, y, z, h='machine', n_list=[0], k_list=[1]):
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
    dG_x = G(x0, y0, z0, x + dx, y, z, omega, n_list=n_list, k_list=k_list) - Green
    dG_y = G(x0, y0, z0, x, y + dy, z, omega, n_list=n_list, k_list=k_list) - Green
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

# lambds = 1e-9
beam = Beam()
beam.E = 4.0            # beam energy in [GeV]
# beam.E = 8.0            # beam energy in [GeV]
beam.I = 0.4 
gamma = beam.E/0.5109906e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)
e_ = 1.60218e-19 #elementary charge

B0 = 1
lperiod = 1# [m] undulator period 

K = 0.9336 * B0 * lperiod * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 
und = Undulator(Kx=K, nperiods=4, lperiod=lperiod, eid="und", phase=0, end_poles='3/4')

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
omega = np.array([THz*1e12*2*np.pi])

xy = 0.1#-0.5e-3#10e-4
x0 = np.linspace(-xy, xy, 41)
y0 = np.linspace(-xy, xy, 41)
z0 = 30

h_erg_s = 1.054571817 * 1e-27 
h_J_s = 1.054571817 * 1e-34 

Int_z = np.array(gamma_z_integral(u_z, z, omega, order=2))
 
# for i in range(1, 100):
i=1
n_list=np.arange(-1, 1, 1)
k_list=np.arange(1, 100, 1)
# n_list = [0]
# k_list = [1]
Green = G(x0, y0, z0, x, y, z, omega, n_list=n_list, k_list=k_list)

nub_G_x, nub_G_y = nubla_G(Green, x0, y0, z0, x, y, z, n_list=n_list, k_list=k_list)
nub_G_x, nub_G_y = nub_G_x[:-1, :, :, :], nub_G_y[:-1, :, :, :]
Green = Green[:-1, :, :, :]

print(np.shape(nub_G_x), np.shape(Green), np.shape(Int_z))

z_dz = np.roll(z, -1)[:-1]
dz = np.array(z_dz - z[:-1])

Gf = lambda ux, Green, Int_z, omega: (4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*ux[:, np.newaxis, np.newaxis, np.newaxis]*Green + nub_G_x) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
# Gf = lambda ux, Green, Int_z, omega: (4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*ux[:, np.newaxis, np.newaxis, np.newaxis]*Green) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])

fG = Gf(ux, Green, Int_z, omega)
# E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis] * (fG), axis=0) #oder 2
E_x = np.sum(dz[:-1, np.newaxis, np.newaxis, np.newaxis] * (fG[:-1] + np.roll(fG, -1,  axis=0)[:-1])/2, axis=0) #oder 3
# E_x = np.sum((f[:-1] + 4* + np.roll(f, -1,  axis=0)[:-1]) * dz[:, np.newaxis, np.newaxis, np.newaxis] / 6, axis=0) #oder 4 to be done

speed_of_light_SGS = speed_of_light * 1e2
A = 0.4 * speed_of_light / (e_ * h_erg_s * 4 * np.pi**2) / (1e7/speed_of_light**2) * 1e-2#* SI unit 1e-4 is due to cm^2 -> mm^2, (1e7/speed_of_light**2) is the factor due to transformation of (I/eh_bar) * c * E**2, I -> I * 1e-1 c, e -> e * 1e-1 c, h_bar -> h_bar * 1e7, c -> c * 1e2, E -> E * 1e6 / c 

I = A * (np.real(E_x[0, 0, :])**2 + np.imag(E_x[0, 0, :])**2)
   

z0 = 30

#%%
# plt.figure('XY, x-pol')
plt.figure()
plt.pcolormesh(x0, y0, np.real(E_x[:,:,0])**2 + np.imag(E_x[:,:,0])**2)
plt.tight_layout()
plt.show()



















