#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:27:42 2022

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

beam = Beam()
beam.E = 4.0            # beam energy in [GeV]
beam.I = 0.4 
# beam.x = 0#180e-6
# beam.xp = 1875e-6
gamma = beam.E/m_e_GeV #0.5109906e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)
e_ = 1.60218e-19 #elementary charge

B0 = 2
lperiod = 0.5# [m] undulator period 
nperiods = 0.01
L_w = lperiod * nperiods
# B0 = 1
# lperiod = 0.5# [m] undulator period 

# K = 0.9336 * B0 * 0.1 * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 
# und = Undulator(Kx=K, nperiods=1.5, lperiod=0.1, eid="und", phase=0, end_poles=1)

K = 0.9336 * B0 * lperiod * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 

drift_b = Undulator(Kx=K*1, nperiods=0.5, lperiod=0.1, eid="und", phase=-np.pi/2, end_poles=0)

# und_st = Undulator(Kx=K/10000/2, nperiods=0.25, lperiod=0.005, eid="und", phase=-np.pi/2, end_poles=0)
und = Undulator(Kx=K*1000/2, nperiods=0.0001, lperiod=1000, eid="und", phase=0, end_poles=0)
# und = Undulator(Kx=K, nperiods=0.5, lperiod=1, eid="und", phase=np.pi/2, end_poles=0)

# und_end = Undulator(Kx=K/10000/2, nperiods=0.25, lperiod=0.005, eid="und", phase=0, end_poles=0)

drift_a = Undulator(Kx=K*0.0000001, nperiods=0.1, lperiod=0.1, eid="und", phase=-np.pi/2, end_poles=0)

# und = Undulator(Kx=K, nperiods=nperiods, lperiod=lperiod, eid="und", phase=0, end_poles=1)
# drift = Drift(l=0.1)#, angle=60*1e-6)
# bend = SBend(l=0.1, k1=1, k2=1)

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
# lat = MagneticLattice((drift_b, und_st, und, und_end, drift_a))
# lat = MagneticLattice((drift_b, und, drift_a))

tau0 = np.copy(p_array.tau())
p_array.tau()[:] = 0
U, E = track4rad_beam(p_array, lat, energy_loss=False, quantum_diff=False, accuracy=1)#

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

def x2xgaus(X, points=3):
    """
    transform coordinates for gauss integration
    | | | | -> |xxx.xxx. ... .xxx.xxx|
    | - coordinate
    x - new coordinate
    . - removed coordinate |

    :param X: array
    :return: new array
    """
    h = X[1] - X[0]    
    if points == 3:
        sqrt35 = 0.5 * np.sqrt(3. / 5.)
        xgaus = np.array([-sqrt35, 0, + sqrt35]) * h
    
    xnew = [X[0]]
    for x in X[:-1]:
        xnew = np.append(xnew, x + xgaus)
    xnew = np.append(xnew, X[-1])
    return xnew
omega = np.linspace(0.001/hr_eV_s, 1500/hr_eV_s, 500) #np.linspace(omega_r*(1-0.9999), omega_r*(1+3), 1000)

z_gaus = x2xgaus(z)
x_gaus = bspline(z, x, z_gaus)
y_gaus = bspline(z, y, z_gaus)
xp_gaus = bspline(z, xp, z_gaus)
yp_gaus = bspline(z, yp, z_gaus)
Bx_gaus = bspline(z, Bx, z_gaus)
By_gaus = bspline(z, By, z_gaus)

ux = xp*beta*speed_of_light
uy = yp*beta*speed_of_light
u_z = np.sqrt((speed_of_light*beta)**2 - ux**2 - uy**2)

ux_gaus = xp_gaus*beta*speed_of_light
uy_gaus = yp_gaus*beta*speed_of_light
u_z_gaus = np.sqrt((speed_of_light*beta)**2 - ux_gaus**2 - uy_gaus**2)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
fig.suptitle('Magnetic field an trajectory', fontsize=24)
fig_size = 12
fig.set_size_inches((2*fig_size,fig_size))

ax1.plot(z, By, linewidth=3)
# ax1.plot(z, Bx, linewidth=3)
# ax1.plot(z_gaus, Bx_gaus, linewidth=3)


# ax1.set_ylim(B0-0.1, B0+0.1)
# plt.plot(z, Bx, linewidth=1)
# plt.plot(z, Bz, linewidth=1)
ax1.set_ylabel(r'$B, [T]$', fontsize=18, labelpad = 0.0)
ax1.grid()

ax2.plot(z, x*1e6, linewidth=3, color='red')
# ax2.plot(z_gaus, x_gaus*1e6, linewidth=3, color='green')
# ax2.plot(z, y*1e6, linewidth=3, color='blue')

ax2.set_ylabel(r'$x, [um]$', fontsize=18, labelpad = 0.0)
ax2.grid()

ax3.plot(z, 1e6*xp, linewidth=3, color='orange')
# ax3.scatter(z_gaus, xp_gaus*1e6, s=0.1, linewidth=3, color='green')
# ax3.plot(z, 1e6*yp, linewidth=3, color='red')

ax3.set_ylabel(r'$xp, [urad]$', fontsize=18, labelpad = 0.0)
ax3.set_xlabel(r'$z, [m]$', fontsize=18, labelpad = 0.0)
ax3.grid()
plt.show()
#%%
z_hat = 10
z0 = z_hat * L_w
z0 = 25

theta_hat = 0
xy=theta_hat*z_hat*(np.sqrt(L_w * lambds/2/np.pi))

screen = Screen()
screen.z = z0     # distance from the begining of lattice to the screen 
screen.start_energy = THz*(1 - 0.9999)*1e12*h_eV_s #0.000001#1e12*h_eV_s    # [eV], starting photon energy
screen.end_energy =   THz*(1 + 3)*1e12*h_eV_s #0.12#10e12*h_eV_s      # [eV], ending photon energy
screen.num_energy = 500
screen.x = -xy # half of screen size in [m] in horizontal plane
screen.y = -xy   # half of screen size in [m] in vertical plane
screen.nx = 1       # number of poinfts in horizontal plane 
screen.ny = 1    # number of points in vertical plane 

screen = calculate_radiation(lat, screen, beam, energy_loss=False, quantum_diff=False, accuracy=1)
data = screen.Total#/max(screen.Total)
X = screen.Eph


ux = xp*beta*speed_of_light
uy = yp*beta*speed_of_light
u_z = np.sqrt((speed_of_light*beta)**2 - ux**2 - uy**2)

# omega_r = 1/h_eV_s #2 * np.pi * speed_of_light / lambds
# omega = np.linspace(0.001/hr_eV_s, 1500/hr_eV_s, 500) #np.linspace(omega_r*(1-0.9999), omega_r*(1+3), 1000)
omega_r = 2 * np.pi * speed_of_light / lambds
omega = np.linspace(omega_r*(1-0.9999), omega_r*(1+3), 500)

# x0 = np.array([0]) #np.linspace(-xy, xy, 1)
# y0 = np.array([0]) #np.linspace(-xy, xy, 1)
x0 = np.linspace(-xy, xy, 1)
y0 = np.linspace(-xy, xy, 1)

#%%
# Gf_x = lambda ux, Green, omega: (4*np.pi*q_e/speed_of_light) * ((1j*omega/speed_of_light**2)*ux[:, np.newaxis, np.newaxis, np.newaxis]*Green) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
# Gf_x(ux, Green, omega)
plt.figure()

# plt.plot(np.cos(integ_beta2(z * 1000, xp)))
# plt.plot(gintegrator_over_traj_py.phase)

# plt.plot(gintegrator_over_traj_py.phaseConstIn)
# plt.plot(gintegrator_over_traj_py.phaseConstCur)

# plt.plot(gintegrator_over_traj_py.arReEx**2 + gintegrator_over_traj_py.arImEx**2)
# plt.plot(gintegrator_over_traj_py.arReEx)
# plt.plot(gintegrator_over_traj_py.arImEx)

# phase = gintegrator_over_traj_py.z - gintegrator_over_traj_py.z[0] + gintegrator_over_traj_py.gamma2*(gintegrator_over_traj_py.XbetaI2)
phase =  gintegrator_over_traj_py.gamma2*(gintegrator_over_traj_py.XbetaI2)

# plt.plot(gintegrator_over_traj_py.radX_array)#, s=1)
# plt.plot(np.cos(phase))
# plt.plot(np.sin(phase))
# plt.plot(phase)


plt.plot(np.real(gintegrator_over_traj_py.radX_array * np.exp(1j*phase)[:-2]))
# plt.plot(np.imag(gintegrator_over_traj_py.radX_array * np.exp(1j*phase)[:-2]))

# plt.plot(np.real(gintegrator_over_traj_py.radX_array))
# plt.plot(np.imag(gintegrator_over_traj_py.radX_array))

# plt.plot(gintegrator_over_traj_py.gamma2*(gintegrator_over_traj_py.XbetaI2))

# plt.plot(gintegrator_over_traj_py.phase)

# plt.plot(gintegrator_over_traj_py.phaseConst)
# plt.plot(gintegrator_over_traj_py.YbetaI2)
# plt.plot(gintegrator_over_traj_py.radX)

# plt.plot(np.cos(gintegrator_over_traj_py.phaseConstIn))
# plt.plot(np.cos(gintegrator_over_traj_py.phaseConstCur))

plt.show()
#%%
# point = 700
# step = np.full((point), 1)
# n = np.arange(0, np.shape(u_z)[0] - point, 1)
# decay_exp = np.exp(-n*0.00004)
# window_func = np.concatenate((step, decay_exp))

Int_z_2 = np.array(gamma_z_integral(u_z, z, omega, order=2))
Int_z_5 = np.array(gamma_z_integral(u_z_gaus, z_gaus, omega, order=5))

# Int_z_5_gaus = np.array([bspline(z[:-1], Int_z_5[:, i], z_gaus) for i in range(0, np.shape(Int_z_5)[1])]).T

fig, (ax1) = plt.subplots(1, num='integral', figsize=(10, 7),)
spectr = -1

ax1.plot(np.real(np.exp(1j*Int_z_5))[:, 1], linewidth=0.7, color='green')#, linestyle=':', alpha=0.4)
# ax1.plot(np.angle(np.exp(1j*Int_z_2))[:, spectr], linewidth=0.7, color='red')#, linestyle=':', alpha=0.4)

# ax1.scatter(z_gaus, np.real(np.exp(1j*Int_z_5_gaus))[:, spectr], s=1, linewidth=0.7, color='green')#, linestyle=':', alpha=0.4)
# ax1.scatter(z_gaus, np.imag(np.exp(1j*Int_z_5_gaus))[:, spectr], s=1, linewidth=0.7, color='red')#, linestyle=':', alpha=0.4)

plt.show()

fig, ax1 = plt.subplots()
ax1.plot(Int_z_5[:, spectr])
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(phase)
plt.show()

#%%
Int_z_5 = np.array(gamma_z_integral(u_z_gaus, z_gaus, omega, order=5))
Int_z_5_gaus = np.array([bspline(z[:-1], Int_z_5[:, i], z_gaus) for i in range(0, np.shape(Int_z_5)[1])]).T

f = Green_func_integrand(x0, y0, z0, x_gaus, y_gaus, z_gaus, omega, ux_gaus, uy_gaus, u_z_gaus, order=5, gradient_term=True, Green_func_type='free_space')

fx = f[0]
fy = f[1]

size = len(z_gaus)
Nmotion = int((size + 1)/3)
a, b = z_gaus[0], z_gaus[-1]

half_step = (b - a) / 2. / (Nmotion - 1)

w = [5/9, 8/9, 5/9]

fx_gaus = fx[1:-1] * np.array(int(size/3)*w)[:, np.newaxis, np.newaxis, np.newaxis]
E_x = np.array(np.sum((fx_gaus[0:-3:3, :, :, :]+fx_gaus[1:-2:3, :, :, :]+fx_gaus[2:-1:3, :, :, :]), axis=0))/1e5 * 5

plt.figure()
plt.plot(np.real(E_x[0,0,:]))
plt.plot(np.imag(E_x[0,0,:]))
plt.show()
#%%
print(np.shape(E_x))
E_y = E_x

# E_x, E_y = Green_func_integrator(np.array(f), z, order=5)
Ix, Iy = SR_field2Intensity(E_x, E_y, I_ebeam=beam.I)
Ix = Ix[0, 0, :] #+ Iy[0,0,:]

def E_ph2THz(x):
    return x / h_eV_s * 1e-12

def THz2E_ph(x):
    return x * h_eV_s / 1e-12

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_title(r"$\hatz$ = " + str(z_hat) + r", $\hat{\theta}$ = " + str(theta_hat))#.format(z_hat, r_hat))
ax1.plot(X/h_eV_s*1e-12, data, linewidth=1, color='black', linestyle='-', label='OCELOT')
ax1.plot(omega*1e-12/2/np.pi, Ix, label=r'Green func', linestyle='-.', linewidth=1, color='blue', alpha=1)

secax = ax1.secondary_xaxis('top', functions=(THz2E_ph, E_ph2THz))
secax.set_xlabel(r'$E_{ph}$, [eV]', fontsize=14)

ax1.set_yscale('log')
# ax1.set_xscale('log')

ax2 = ax1.twinx() 
ax2.plot(X/h_eV_s*1e-12, data, linewidth=1, color='black', linestyle='-', label='OCELOT')
ax2.plot(omega*1e-12/2/np.pi, Ix, label=r'Green func', linestyle='-.', linewidth=1, color='blue', alpha=1)
ax2.set_yticks([])

# ax3 = ax1.twinx() 
# ax3.plot(omega*1e-12/2/np.pi, np.where(abs((data - Ix)/(data))*1e2<=2, (data - Ix)/(data)*1e2, None), label=r'Norm. discrepancy', linewidth=1, color='red', alpha=1)
# # ax3.plot(omega*1e-12/2/np.pi, (data - I)/(data)*1e2, label=r'Norm. discrepancy', linewidth=0.5, color='red', alpha=1)

# # ax3.set_yscale('log')
# ax3.set_ylabel('($I_{Green}$ - $I_{oclot}$)/$I_{Green}$, $\%$', fontsize=14, color='red')
# ax3.tick_params(axis='y', colors='red') 
secax3 = ax2.secondary_yaxis(-0.13)
secax3.set_ylabel('Flux, arb.units', fontsize=14)

# ax1.set_ylim(1e4)
# ax1.set_xlim(0, np.max(omega*1e-12/2/np.pi))
ax2.set_ylim(0)
# ax2.set_xlim(0, np.max(omega*1e-12/2/np.pi))

ax1.legend(loc=3)
# ax3.legend(loc=4)

ax1.set_xlabel(r'$\Omega$, [THz]', fontsize=14)
# ax1.set_xlabel('E_ph, eV', fontsize=14)
ax1.grid()
plt.tight_layout()
plt.show()
#%%
ux = xp[:-1]*beta*speed_of_light
uy = yp[:-1]*beta*speed_of_light
u_z = np.sqrt((speed_of_light*beta)**2 - ux**2 - uy**2)

# gamma_z_2 = np.array(gamma_z_integral(u_z, z, omega, order=2))
# gamma_z_3 = np.array(gamma_z_integral(u_z, z, omega, order=3))
# gamma_z_4 = np.array(gamma_z_integral(u_z, z, omega, order=4))

# from scipy.fft import fft, ifft
# from scipy.fft import fft, fftfreq, fftshift
 
# gamma_z_2_fft = fft(gamma_z_2[:, -1])

# gamma_z_2_fft = fft(np.exp(1j*gamma_z_2[:, -1]))
# gamma_z_2_fft = fft(1j*gamma_z_2[:, -1])

fig, (ax1, ax2) = plt.subplots(2, 1, num='integral', figsize=(10, 7),)

# fig, ax1 = plt.subplots()# plt.plot(np.real(np.exp(1j*gamma_z[-1][:])))
# plt.plot((gamma_z[0:3][0:1]))
# ax1.plot(z[1:], gamma_z)
# ax1.plot(z[1:], np.real(np.exp(1j*np.array(gamma_z[-1][:]))))
# ax1.plot(np.real(np.exp(1j*gamma_z_2[:, -1])), linewidth=0.7, linestyle='-.', color='red', alpha=0.4)
# ax1.plot(np.imag(np.exp(1j*gamma_z_2[:, -1])), linewidth=0.7, color='green', linestyle='-.', alpha=0.4)

# ax1.plot(np.real(np.array(G)[:, 0, 0, 0]), linewidth=0.7, color='green', linestyle=':')#, alpha=0.4)
# ax1.plot(np.imag(np.array(G)[:, 0, 0, 0]), linewidth=0.7, color='red', linestyle=':')#, alpha=0.4)

# point = 600
# step = np.full((point), 1)
# n = np.arange(0, np.shape(u_z)[0] - point, 1)
# decay_exp = np.exp(-n*0.02)

# window_func = np.concatenate((step, decay_exp))

spectr = -1
ax1.plot(np.real(np.array(f)[0, :, 0, 0,spectr]), linewidth=0.7, color='green', linestyle=':', alpha=0.4)
ax1.plot(np.imag(np.array(f)[0, :, 0, 0, spectr]), linewidth=0.7, color='red', linestyle=':', alpha=0.4)
ax1.plot(abs(np.array(f)[0, :, 0, 0, spectr]), linewidth=0.7, color='black', linestyle=':', alpha=0.4)

ax1.set_xlim(0)
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.plot(np.real(np.exp(1j*gamma_z_3[:, -1])))
# ax1.plot(np.real(np.exp(1j*gamma_z_4[:, -1])))

# ax2 = ax1.twinx()


Integral = np.array([])
for i in range(np.shape(f)[1]):
    Integral = np.append(Integral, np.sum(np.array(f)[0, 0:i, 0, 0, spectr]))
    
ax2.plot(np.real(Integral), color='green', linewidth=0.7)
ax2.plot(np.imag(Integral), color='red', linewidth=0.7)
ax2.plot(abs(np.array(Integral)), linewidth=0.7, color='black', linestyle='-.', alpha=0.4)

# ax2.plot(np.real(gamma_z_2_fft), color='red')
# ax2.plot(np.imag(gamma_z_2_fft), color='green')

# ax2.plot(np.sqrt(np.real(gamma_z_2_fft)**2 + np.imag(gamma_z_2_fft)**2), color='black')
ax2.set_xlim(0)
# ax2.set_yscale('log')
# ax2.set_xscale('log')

# ax2.plot(z[1:], u_z)

plt.show()
#%%
# THz = 24.5
E_ph = 500
THz =  E_ph2THz(E_ph)#speed_of_light * 1e-12 / lambds#0.000299792458/lambds
print(THz)

omega = np.array([THz*1e12 * 2* np.pi])
print(E_ph2THz(omega))

z_hat = 10
z0 = z_hat * L_w

theta_hat_x = 0.01
theta_hat_y = 0.001

x_lim=theta_hat_x*z_hat*(np.sqrt(L_w * lambds/2/np.pi))
y_lim=theta_hat_y*z_hat*(np.sqrt(L_w * lambds/2/np.pi))

n =101
x0 = np.linspace(-x_lim, x_lim, n)
y0 = np.linspace(-y_lim, y_lim, n)

# m_list = [1]
# k_0 = 3000
# k_list=np.arange(k_0, k_0 + 10, 1)
# pipe={'R' : 0.1, 'm_list' : m_list, 'k_list' : k_list}

# n_list = [0]
# k_0 = 1
# k_list=np.arange(k_0, k_0 + 3000, 1)
# iris={'a' : 0.04, 'b' : 0.01, 'n_list' : n_list, 'k_list' : k_list}

f = Green_func_integrand(x0, y0, z0, x, y, z, omega, ux, uy, u_z, order=3, gradient_term=False, Green_func_type='free_space')#, **pipe)
E_x, E_y = Green_func_integarator(f*window_func[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis], z, order=3) 
Ix, Iy = SR_field2Intensity(E_x, E_y, out="Components", I_ebeam=beam.I)#[0,0,:]
# Ix = Iy
screen = Screen()
screen.z = z0      # distance from the begining of lattice to the screen 
screen.size_x = x_lim # half of screen size in [m] in horizontal plane
screen.size_y = y_lim   # half of screen size in [m] in vertical plane
screen.nx = n       # number of poinfts in horizontal plane 
screen.ny = n    # number of points in vertical plane 

# X/h_eV_s*1e-12
screen.start_energy = E_ph#THz*1e12*h_eV_s# [eV], starting photon energy
screen.end_energy =   E_ph#THz*1e12*h_eV_s     # [eV], ending photon energy
screen.num_energy = 1  
screen = calculate_radiation(lat, screen, beam, energy_loss=True, quantum_diff=True, accuracy=0.5)

data = screen.Total

dfl = RadiationField()
dfl = screen2dfl(screen, polarization='x')
constQuant = 3*alpha/(q_e*1e2)/(4*pi**2)*1e-3 * beam.I * gamma * gamma / (z0*1e2)**2
I_oclot= dfl.intensity() * constQuant


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, num='2D', figsize=(15, 5))
fig.suptitle("{} THz, {} eV".format(round(THz), round(THz*1e12*h_eV_s, 2)) + r'$, \hat{z}$ = ' + str(z_hat), fontsize=16)

theta_x = x0 / (z_hat*(np.sqrt(L_w * lambds/2/np.pi)))
theta_y = y0 / (z_hat*(np.sqrt(L_w * lambds/2/np.pi)))

ax1.pcolormesh(dfl.scale_x()/(z_hat*(np.sqrt(L_w * lambds/2/np.pi))), dfl.scale_y()/(z_hat*(np.sqrt(L_w * lambds/2/np.pi))), I_oclot[0, :, :])
ax1.set_title("OCELOT")
# ax1.set_aspect('equal') 

ax2.pcolormesh(theta_x, theta_y, Ix[:, :, 0])
ax2.set_title("Green func")
ax2.set_yticks([])
# ax2.set_aspect('equal') 

im = ax3.pcolormesh(theta_x, theta_y, (Ix[:, :, 0] - I_oclot[0, :, :])/np.max(Ix) *1e2, norm=colors.CenteredNorm(), cmap='seismic')
ax3.set_title("Discrepancy")
cax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
fig.colorbar(im, cax=cax, use_gridspec=0)
cax.set_ylabel('($I_{Green}$ - $I_{oclot}$)/max($I_{Green}$), $\%$', fontsize=16)
ax3.set_yticks([])
# ax3.set_aspect('equal') 

ax1.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
ax2.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
ax1.set_ylabel(r'$\hat{\theta}_y$', fontsize=14)
ax3.set_xlabel(r'$\hat{\theta}_x$', fontsize=14)
# plt.tight_layout()
# ax2.set_ylabel('y, [m]')
plt.show()
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, num='1D', figsize=(10, 5),)
fig.suptitle("Slices", fontsize=16)

ax1.plot(dfl.scale_x()/(z_hat*(np.sqrt(L_w * lambds/2/np.pi))), I_oclot[0, n//2, :], c='black', label="OCELOT")
ax1.plot(theta_x, Ix[n//2, :, 0], '-.', c='blue', label="Green func")
ax1.set_ylim(0)

# ax1_twin = ax1.twinx() 
# ax1_twin.plot(theta_y, (Ix[n//2, :, 0] - I_oclot[0, n//2, :])/np.max(Ix) *1e2, label=r'Norm. discrepancy', linewidth=0.75, color='red', alpha=1)
# ax1_twin.set_ylabel('Discrepancy, $\%$', fontsize=16, c='red')
# ax1_twin.tick_params(axis='y', colors='red') 

# ax1_twin.set_yscale('log')

ax2.plot(dfl.scale_y()/(z_hat*(np.sqrt(L_w * lambds/2/np.pi))), I_oclot[0, :, n//2], c='black', label="OCELOT")
ax2.plot(theta_y, Ix[:, n//2, 0], '-.', c='blue', label="Green func")
ax2.set_ylim(0)

# ax2_twin = ax2.twinx() 
# ax2_twin.plot(theta_y, (Ix[:, n//2, 0] - I_oclot[0, :, n//2])/np.max(Ix) *1e2, label=r'Norm. discrepancy', linewidth=0.75, color='red', alpha=1)
# ax2_twin.set_ylabel('Discrepancy, $\%$', fontsize=16, c='red')
# ax2_twin.tick_params(axis='y', colors='red') 

# ax2_twin.set_yscale('log')


ax1.set_xlabel('x, [m]', fontsize=14)
ax2.set_xlabel('y, [m]', fontsize=14)
ax1.set_ylabel('Flux', fontsize=14)
ax2.set_ylabel('Flux', fontsize=14)


ax1.legend()
ax1_twin.legend()
plt.tight_layout()
plt.show()