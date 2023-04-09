#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:57:10 2023

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
from ocelot.cpbd.elements import Undulator
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.beam import *
from ocelot.rad.screen import Screen
from ocelot.rad.radiation_py import calculate_radiation, track4rad_beam, traj2motion
from ocelot.optics.wave import dfl_waistscan, screen2dfl, RadiationField
from ocelot.gui.dfl_plot import plot_dfl, plot_dfl_waistscan

from ocelot.common.ocelog import *
ocelog.setLevel(logging.DEBUG)
_logger = logging.getLogger(__name__)

from SRinWguide import *

def E_ph2THz(x):
    return x / h_eV_s * 1e-12

def THz2E_ph(x):
    return x * h_eV_s / 1e-12
    
def track2motion(beam, lat, energy_loss=True, quantum_diff=True):
    
    if beam.__class__ is Beam:
        p = Particle(x=beam.x, y=beam.y, px=beam.xp, py=beam.yp, E=beam.E)
        p_array = ParticleArray()
        p_array.list2array([p])
    
    # lat = MagneticLattice(und)
    tau0 = np.copy(p_array.tau())
    p_array.tau()[:] = 0
    U, E = track4rad_beam(p_array, lat, energy_loss=False, quantum_diff=False, accuracy=1)#
    U = np.concatenate(U)
    U = np.concatenate(U)
    
    motion = Motion()
    
    motion.x =  U[0::9]
    motion.y =  U[2::9]
    motion.z =  U[4::9]
    motion.bx = U[1::9]
    motion.by = U[3::9]
    motion.bz = U[5::9]
    motion.Bx = U[6::9]
    motion.By = U[7::9]
    motion.Bz = U[8::9]
    
    return motion

def SR_norm_on_ebeam_I(I_ebeam=0, norm='e_beam'):
    '''
    Parameters
    ----------
    Ex : array
        Ex component of the field.
    Ey : array
        Ey component of the field.
    out : str, optional
        Which intensity to return, of the given component or full component. The default is "Components".
    norm : str, optional
        DESCRIPTION. The default is 'e_beam'. Normalisation on electron beam current.
        
    Returns
    -------
    Intensity.

    '''
    print('\n Normalizing radiation field...')

    h_erg_s = 1.054571817 * 1e-27 
    h_J_s = 1.054571817 * 1e-34 
    
    if norm=='e_beam':
        speed_of_light_SGS = speed_of_light * 1e2
        A = I_ebeam * speed_of_light / (q_e* h_erg_s * 4 * np.pi**2) / (1e7/speed_of_light**2) * 1e-2#* SI unit 1e-4 is due to cm^2 -> mm^2, (1e7/speed_of_light**2) is the factor due to transformation of (I/eh_bar) * c * E**2, I -> I * 1e-1 c, e -> e * 1e-1 c, h_bar -> h_bar * 1e7, c -> c * 1e2, E -> E * 1e6 / c 
 
    return A


def generate_SR_ocelot_dfl(lat, beam, z0, shape=(51, 51, 100), dgrid_xy=(1e-3, 1e-3), E_ph=np.array([1, 1000]),
                    polarization='x',
                    energy_loss=False, quantum_diff=False):

    screen = Screen()
    screen.z = z0     # distance from the begining of lattice to the screen 
    
    if np.size(E_ph) == 1: 
        E_ph = np.array([E_ph, E_ph])
        print(E_ph)  
        
    screen.start_energy = E_ph[0] # [eV], starting photon energy
    screen.end_energy =   E_ph[1] # [eV], ending photon energy
    screen.num_energy = shape[2]
    
    # screen.x = dgrid_xy[0] # half of screen size in [m] in horizontal plane
    # screen.y = dgrid_xy[1] # half of screen size in [m] in vertical plane
    screen.size_x = dgrid_xy[0]/2 # half of screen size in [m] in horizontal plane
    screen.size_y = dgrid_xy[1]/2 # half of screen size in [m] in vertical plane
    screen.nx = shape[0] # number of points in horizontal plane 
    screen.ny = shape[1] # number of points in vertical plane 

    screen = calculate_radiation(lat, screen, beam, energy_loss=energy_loss, quantum_diff=quantum_diff, accuracy=1)
    
    dfl = RadiationField()
    dfl = screen2dfl(screen, polarization=polarization, norm='ebeam', beam=beam)

    return dfl

def generate_SR_Green_dfl(lat, beam, z0, shape=(51, 51, 100), dgrid_xy=(1e-3, 1e-3), E_ph = np.array([1, 1000]),
                    gradient_term=True, order=5, Green_func_type='free_space', polarization='x',
                    energy_loss=False, quantum_diff=False):

    motion = track2motion(beam, lat, energy_loss=energy_loss, quantum_diff=quantum_diff)
    
    gamma = beam.E/0.51099890221e-03 #relative energy E_electron/m_e [GeV/Gev]
    beta = np.sqrt(1 - 1/gamma**2)
 
    dfl = RadiationField((shape[2], shape[1], shape[0]))
    
    dfl.domain_z = 'f'
    dfl.domain_xy = 's'
    dfl.dx = dgrid_xy[0] / dfl.Nx()
    dfl.dy = dgrid_xy[1] / dfl.Ny()
    
    if dfl.Nz() != 1:
        E_ph = np.linspace(E_ph[0], E_ph[1], dfl.Nz())
        scale_kz = 2 * np.pi * E_ph / h_eV_s / speed_of_light
        k = (scale_kz[-1] + scale_kz[0]) / 2
        dk = (scale_kz[-1] - scale_kz[0]) / dfl.Nz()
        dfl.dz = 2 * np.pi / dk / dfl.Nz()
        dfl.xlamds = 2 * np.pi / k
        omega = E_ph / h_eV_s * 2 * np.pi
    else:
        dfl.dz = 1
        dfl.xlamds = h_eV_s * speed_of_light / E_ph
        omega = np.array([E_ph / h_eV_s * 2 * np.pi])

    ux = motion.bx*beta*speed_of_light
    uy = motion.by*beta*speed_of_light
    u_z = np.sqrt((speed_of_light*beta)**2 - ux**2 - uy**2)
    
    f = Green_func_integrand(dfl.scale_x(), dfl.scale_y(), z0, motion.x, motion.y, motion.z, omega, 
                             ux, uy, u_z, order=order, 
                             gradient_term=gradient_term, Green_func_type=Green_func_type)
    
    E_x, E_y = Green_func_integrator(f, motion.z, order=order)
    
    A = SR_norm_on_ebeam_I(I_ebeam=beam.I)
    E_x, E_y = E_x * np.sqrt(A), E_y * np.sqrt(A)
    
    if polarization == 'x':
        dfl.fld = np.swapaxes(E_x, 0, 2) 
    elif polarization == 'y':
        dfl.fld = np.swapaxes(E_y, 0, 2) 
        
    return dfl


def plot_motion(motion, direction='x'):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    # fig.suptitle('Magnetic field and trajectory', fontsize=24)
    fig_size = 8
    fig.set_size_inches((2*fig_size,fig_size))
    
    if direction=='x':
        magnetic_field = motion.By
        magnetic_field_label = r'$B_y, [T]$'
        traj = motion.x*1e6
        traj_label = r'$x, [um]$'
        beta = motion.bx*1e6
        beta_label = r'$\beta_x \times 10^6, [urad]$'
    elif direction=='y':
        magnetic_field = motion.Bx
        magnetic_field_ylabel = r'$B_x, [T]$'
        traj = motion.y*1e6
        traj_label = r'$y, [um]$'
        beta = motion.by*1e6
        beta_label = r'$\beta_x \times 10^6, [urad]$'

    ax1.plot(motion.z, magnetic_field, linewidth=2)

    ax1.set_ylabel(magnetic_field_label, fontsize=18, labelpad = 0.0)
    ax1.grid()
    
    ax2.plot(motion.z, traj, linewidth=2, color='red')
    
    ax2.set_ylabel(traj_label, fontsize=18, labelpad = 0.0)
    ax2.grid()
    
    ax3.plot(motion.z, beta, linewidth=2, color='orange')
    
    ax3.set_ylabel(beta_label, fontsize=18, labelpad = 0.0)
    ax3.set_xlabel(r'$z, [m]$', fontsize=18, labelpad = 0.0)
    ax3.grid()
    '''
    this is the a part to calculate the phase
    #the following lines calculate the phase advance in the magnetic line. This is not important, please, ignore it if you want.
    n = np.size(motion.z)
    h = (np.max(motion.z) - np.min(motion.z))/n/1e3
    # a = [1,2,3,4,5,6,7,8,9]
    x_prime = np.array([])
    phi = np.array([])
    x_prime = (q_e/(speed_of_light*m_e_kg*gamma))*np.append(x_prime, [h*np.sum(motion.By[:i-1]) for i in range(1, n+1)])

    # f = np.square(xp)
    phi1 = np.array([])
    phi2 = np.array([])

    # f2 = np.square(xp)
    f2 = np.square(motion.bx)

    h_i = (motion.z - np.roll(motion.z, 1))

    for i in range(1, n):
        # I2 = np.sum(h_i[0:i]*f2[0:i])/1e3
        I2 = np.sum((h_i[1:i+1])*(f2[0:i] + f2[1:i+1])/2)/1e3
        # I2 = np.sum((h_i[2:2*i])*(f2[0:2*i-2] + 4*f2[1:2*i-1] + f2[2:2*i])/6)/1e3
        phi2 = np.append(phi2, I2)
        
    phi2 = (1/(2*xlambs))*phi2

    # ax4_copy=ax4.twinx()
    # ax4.plot(z, phi1, linewidth=3, color='green')
    # ax4_copy.plot(z, f2, linewidth=3, color='blue')
    z = motion.z[1:]
    phi2 = z*1e-3/(xlambs*2*gamma**2) + phi2
    
    ax4.plot(z, phi2, linewidth=2, color='green')
    ax4.set_ylabel(r'$\phi/2\pi$', fontsize=18, labelpad = 0.0)
    ax4.set_xlabel(r'$z, [mm]$', fontsize=18, labelpad = 0.0)

    ax4.grid()
    '''
    plt.show()
    
def plot_2_dfl_spectra(dfl1, dfl2, show_THz=False, show_discrepancy=False, show_logy=False, z_hat=None, theta_hat=None):
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    if None not in [z_hat, theta_hat]:
        ax1.set_title(r"$\hat$ = " + str(z_hat) + r", $\hat{\theta}$ = " + str(theta_hat))

    ax1.plot(dfl1.phen(), dfl2.intensity()[:, 0, 0], linewidth=1, color='black', linestyle='-', label='OCELOT')
    ax1.plot(dfl2.phen(), dfl2.intensity()[:, 0, 0], label=r'Green func', linestyle='-.', linewidth=1, color='blue', alpha=1)
    
    ax1.set_xlabel(r'$E_{ph}$, [eV]', fontsize=14)
    ax1.set_xlim(np.min(dfl2.phen()), np.max(dfl2.phen()))
    ax1.set_ylim(0)

    ax1.legend(loc=3)
    ax1.grid()

    if show_THz:
        secax = ax1.secondary_xaxis('top', functions=(E_ph2THz, THz2E_ph))
        # secax = ax1.secondary_xaxis('top', functions=(THz2E_ph, E_ph2THz))
        secax.set_xlabel(r'$\Omega$, [THz]', fontsize=14)
        # secax.set_xlabel(r'$E_{ph}$, [eV]', fontsize=14)
# 
    if show_logy:
        ax2 = ax1.twinx() 
        ax2.plot(dfl1.phen(), dfl1.intensity()[:, 0, 0], linewidth=1, color='black', linestyle='-', label='OCELOT')
        ax2.plot(dfl2.phen(), dfl2.intensity()[:, 0, 0], label=r'Green func', linestyle='-.', linewidth=1, color='blue', alpha=1)
        ax2.set_yscale('log')
        ax2.set_yticks([])
        ax2.set_ylim(0)
    
    if show_discrepancy:
        ax3 = ax1.twinx() 
        I_1 = dfl1.intensity()[:, 0, 0]
        I_2 = dfl2.intensity()[:, 0, 0]
    
        ax3.plot(dfl_ocelot.phen(), 
                 np.where(abs((I_1 - I_2)/(I_1))*0.5e2<=2, 
                          ((I_1 - I_2)/I_1)*1e2, None), label=r'Norm. discrepancy', linewidth=0.5, color='red', alpha=1)
    

        ax3.set_ylim(-5, 5)
        ax3.set_ylabel('($I_{ocelot}$ - $I_{Green}$)/$I_{ocelot}$, $\%$', fontsize=14, color='red')
        ax3.tick_params(axis='y', colors='red') 
        secax3 = ax2.secondary_yaxis(-0.07)
        secax3.set_ylabel('Flux, arb.units', fontsize=14)
        ax3.legend(loc=4)
        
    plt.tight_layout()
    plt.show()
    

#%%
    
# ebeam parameters
beam = Beam()
beam.E = 4.0            # beam energy in [GeV]
beam.I = 0.4 
gamma = beam.E/0.51099890221e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)

# undulator parameters
B0 = 0.4
lperiod = 0.03# [m] undulator period 
nperiods = 15
L_w = lperiod * nperiods
K = 0.9336 * B0 * lperiod * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 

und = Undulator(Kx=K, nperiods=nperiods, lperiod=lperiod, eid="und", end_poles='1')

# radiation parameters
lambds = (lperiod/2/gamma**2)*(1 + K**2/2)
print('E_ph = ', round(speed_of_light * h_eV_s / lambds, 4), 'eV')
THz =  speed_of_light * 1e-12 / lambds 
print('omega = ', round(THz,1), 'THZ')
lat = MagneticLattice(und)

motion = track2motion(beam, lat, energy_loss=0, quantum_diff=0)
plot_motion(motion)

#%%
z_hat = 10
z0 = z_hat * L_w

theta_hat = 0
xy=theta_hat*z_hat*(np.sqrt(L_w * lambds/2/np.pi))
#%%

dfl_Green = generate_SR_Green_dfl(lat, beam, z0, shape=(1, 1, 1000), dgrid_xy=(0, 0), E_ph=np.array([0.311, 12447]),
                      gradient_term=True, order=5, Green_func_type='free_space', polarization='x')

dfl_ocelot = generate_SR_ocelot_dfl(lat, beam, z0, shape=(1, 1, 1000), dgrid_xy=(0, 0), E_ph=np.array([0.311, 12447]),
                                    polarization='x')

plot_2_dfl_spectra(dfl_ocelot, dfl_Green, show_THz=True, show_discrepancy=1, show_logy=1)
#%%

dfl_Green = generate_SR_Green_dfl(lat, beam, z0, shape=(101, 101, 1), dgrid_xy=(1e-3, 1e-3), E_ph = 3138,
                      gradient_term=1, order=3, Green_func_type='free_space', polarization='x')


dfl_ocelot = generate_SR_ocelot_dfl(lat, beam, z0, shape=(101, 101, 1), dgrid_xy=(1e-3, 1e-3), E_ph = 3138,
                                    polarization='x')

plot_dfl(dfl_ocelot, fig_name='dfl_ocelot')
plot_dfl(dfl_Green, fig_name='dfl_Green')

#%%
def plot_2_dfl_2D_dist(dfl1, dfl2, z_hat=None, L_w=None, dfl1_label='Green', dfl2_label='ocelot'):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, num='2D', figsize=(15, 5))
    
    if z_hat is not None:
        fig.suptitle("{} THz, {} eV".format(round(dfl1.phen()[0]/1e12/h_eV_s, 2), round(dfl1.phen()[0])) + r'$, \hat{z}$ = ' + str(z_hat), fontsize=16)
    
    dfl1.to_domain('sf')
    
    try:
        norm = z_hat*(np.sqrt(L_w * lambds/2/np.pi))
        theta1_x = dfl1.scale_x() / norm
        theta1_y = dfl1.scale_y() / norm
        theta2_x = dfl2.scale_x() / norm
        theta2_y = dfl2.scale_y() / norm
    except TypeError:
        theta1_x = dfl1.scale_x() * 1e6
        theta1_y = dfl1.scale_y() * 1e6
        theta2_x = dfl2.scale_x() * 1e6
        theta2_y = dfl2.scale_y() * 1e6
        
    I1 = dfl1.intensity()[0, :, :]
    I2 = dfl2.intensity()[0, :, :] 
     
    ax1.pcolormesh(theta1_x, theta1_y, I1)
    ax1.set_title(dfl2_label)
    ax1.set_aspect('equal') 
    
    ax2.pcolormesh(theta2_x, theta2_y, I2)
    ax2.set_title(dfl1_label)
    ax2.set_yticks([])
    ax2.set_aspect('equal') 
    
    
    im = ax3.pcolormesh(theta2_x, theta2_y, (I2 - I1)/np.max(I2)*1e2, norm=colors.CenteredNorm(), cmap='seismic')
    ax3.set_title("Discrepancy")
    cax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
    fig.colorbar(im, cax=cax, use_gridspec=0)
    cax.set_ylabel('($I_2$ - $I_1$)/max($I_2$), $\%$', fontsize=16)
    ax3.set_yticks([])
    ax3.set_aspect('equal') 
    
    ax1.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
    ax2.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
    ax1.set_ylabel(r'$\hat{\theta}_y$', fontsize=14)
    ax3.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
    # plt.tight_layout()
    # ax2.set_ylabel('y, [m]')
    plt.show()
    
plot_2_dfl_2D_dist(dfl_Green, dfl_ocelot, z_hat=z_hat, L_w=L_w, dfl1_label='Green', dfl2_label='Ocelot')


def plot_2_dfl_1D_dist(dfl1, dfl2, z_hat=None, L_w=None, dfl1_label='Green', dfl2_label='Ocelot'):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, num='1D', figsize=(10, 5),)
    # fig.suptitle("Slices", fontsize=16)
    
    I1 = dfl1.intensity()[0, :, :]
    I2 = dfl2.intensity()[0, :, :]
    
    try:
        norm = z_hat*(np.sqrt(L_w * lambds/2/np.pi))
        theta1_x = dfl1.scale_x() / norm
        theta1_y = dfl1.scale_y() / norm
        theta2_x = dfl2.scale_x() / norm
        theta2_y = dfl2.scale_y() / norm
    except TypeError:
        theta1_x = dfl1.scale_x() * 1e6
        theta1_y = dfl1.scale_y() * 1e6
        theta2_x = dfl2.scale_x() * 1e6
        theta2_y = dfl2.scale_y() * 1e6
    
    ax1.plot(theta1_x, I1[dfl1.Ny()//2 + 1, :], c='black', label=dfl1_label)
    ax1.plot(theta2_x, I2[dfl2.Ny()//2 + 1, :], '-.', c='blue', label=dfl2_label)
    ax1.set_ylim(0)
    
    ax1_twin = ax1.twinx() 
    ax1_twin.plot(theta1_x, (I1[dfl1.Ny()//2 + 1, :] - I2[dfl2.Ny()//2 + 1, :])/np.max(I1) *1e2, label=r'D', linewidth=0.75, color='red', alpha=1)
    ax1_twin.set_ylabel('($I_2$ - $I_1$)/max($I_2$), $\%$', fontsize=16, c='red')
    ax1_twin.tick_params(axis='y', colors='red') 
    
    # ax1_twin.set_yscale('log')
    
    ax2.plot(theta1_y, I1[:, dfl1.Nx()//2 + 1], c='black', label=dfl1_label)
    ax2.plot(theta2_y, I2[:, dfl2.Nx()//2 + 1], '-.', c='blue', label=dfl2_label)
    ax2.set_ylim(0)
    
    ax2_twin = ax2.twinx() 
    ax2_twin.plot(theta1_y, (I1[:, dfl1.Nx()//2 + 1] - I2[:, dfl2.Nx()//2 + 1])/np.max(I1) *1e2, label=r'D', linewidth=0.75, color='red', alpha=1)
    ax2_twin.set_ylabel('($I_2$ - $I_1$)/max($I_2$), $\%$', fontsize=16, c='red')
    ax2_twin.tick_params(axis='y', colors='red') 
    
    # ax2_twin.set_yscale('log')
    
    ax1.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
    ax2.set_xlabel(r'$\hat{\theta}_y$', fontsize=14)
    ax1.set_ylabel('Flux', fontsize=14)
    ax2.set_ylabel('Flux', fontsize=14)

    ax1.legend(loc=2, fontsize=14)
    ax2_twin.legend(loc=4, fontsize=14)
    plt.tight_layout()
    plt.show()
    
plot_2_dfl_1D_dist(dfl_Green, dfl_ocelot, z_hat=z_hat, L_w=L_w, dfl1_label='Green', dfl2_label='Ocelot')
    
    
    













