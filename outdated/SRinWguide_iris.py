#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 12:35:36 2022

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

def G_free_space(x0, y0, z0, x, y, z, omega):
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

def G_iris(x0, y0, z0, x, y, z, omega, a=0.1, b=0.1, n_list=[0], k_list=[1]):
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
    print('    calculating G_iris...')
    start = time.time()
    print('    n =', a, 'k =', b, 'n_list =', n_list, 'k_list =', k_list)

    x0, y0 = np.meshgrid(x0, y0)
    
    phi0 = np.arctan2(y0, x0)
    phi = np.arctan2(y, x)
    
    r0 = np.sqrt(x0**2 + y0**2)
    r =  np.sqrt(x**2 + y**2)
    
    beta0 = 0.824 
    delta = (1 + 1j) * beta0 * np.sqrt((speed_of_light * b)/(4 * omega * a**2))
    
    Green = np.zeros((np.shape(z)[0], np.shape(x0)[0], np.shape(y0)[0], np.shape(omega)[0]))
    G_nk_array = []
    for nk in itertools.product(n_list, k_list):
        n = nk[0]
        k = nk[1]
        v_nk = special.jn_zeros(n, k)[-1]
        print('    n =', n, 'k =', k, 'v_nk =', v_nk)
        
        G_nk = -((1j * speed_of_light * (1.0 - 2.0*delta[np.newaxis, np.newaxis, np.newaxis, :])) / (2 * np.pi * omega[np.newaxis, np.newaxis, np.newaxis, :] * a**2)) * \
            (np.exp(-1j*n*(phi0[np.newaxis, :, :, np.newaxis] - phi[:, np.newaxis, np.newaxis, np.newaxis]))) / (special.jv(n+1, v_nk))**2 * \
                np.exp(-(1j*speed_of_light*(z0 - z[:,np.newaxis,np.newaxis,np.newaxis]) * v_nk**2 * (1.0 - 2.0*delta[np.newaxis,np.newaxis,np.newaxis,:])) / (2 * omega[np.newaxis,np.newaxis,np.newaxis,:] * a**2)) * \
                    special.jv(n, v_nk * (1 - delta[np.newaxis,np.newaxis,np.newaxis,:])*r[:, np.newaxis, np.newaxis, np.newaxis]/a) * \
                        special.jv(n, v_nk * (1 - delta[np.newaxis,np.newaxis,np.newaxis,:])*r0[np.newaxis, :, :, np.newaxis]/a)
        
        Green = np.add(Green, G_nk)  #still summing problem here! or not!? 

    t_func = time.time() - start
    print('        calculated G_iris in %.2f ' % t_func + 'sec')
        
    return np.array(Green)

def nubla_G(Green, x0, y0, z0, x, y, z, h='machine', Green_func_type='free_space', **kwargs):
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
    print('    calculating nubla_G...')
    start = time.time()

    epsilon=1.0
    while epsilon+1>1:
        epsilon=epsilon/2
    epsilon=epsilon*1e2
    print("The value of epsilon is: ", epsilon)
    
    if h=='machine':
        bool_x = np.where(x!=0, True, False)
        dx = np.where(x!=0, np.sqrt(epsilon)*abs(x), np.sqrt(epsilon)*1)
        dy = np.where(y!=0, np.sqrt(epsilon)*abs(y), np.sqrt(epsilon)*1)
    elif isinstance(dx, float):
        dx = h
        dy = h
    else: 
        ValueError('please, enter a float "h" for taking derivative')
    
    if Green_func_type == 'free_space':
        dG_x = G_free_space(x0, y0, z0, x + dx, y, z, omega) - Green
        dG_y = G_free_space(x0, y0, z0, x, y + dy, z, omega) - Green
    elif Green_func_type == 'iris':
        dG_x = G_iris(x0, y0, z0, x + dx, y, z, omega, **kwargs) - Green
        dG_y = G_iris(x0, y0, z0, x, y + dy, z, omega, **kwargs) - Green
        
    t_func = time.time() - start
    print('        calculated nubla_G in %.2f ' % t_func + 'sec')

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

f_gamma = lambda u_z, omega: np.array((omega[np.newaxis, :]/2/speed_of_light)*(1/gamma_z(u_z)[:, np.newaxis]**2))
def gamma_z_integral(u_z, z, omega, order=2):
    print('    taking gamma_z_integral...')
    start = time.time()
    
    I_array = np.array([])
    z_dz = np.roll(z, -1)[:-1]
    dz = np.array(z_dz - z[:-1])
    
    func = f_gamma(u_z, omega)
    
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
    
    t_func = time.time() - start
    print('        gamma_z_integral is taken in %.2f ' % t_func + 'sec')

    return I_array

def Green_func_integrand(x0, y0, z0, x, y, z, omega, ux, uy, u_z, order=3, gradient_term=False, Green_func_type='free_space', **kwargs):
    '''
    Parameters
    ----------
    z0 : array
        the longitudinal position at which the field will be calculated.
    x0, y0 : array
        transverse position at which the field will be calculated.

    z : array
        integration position along the electron trajectory.
    x : array
        transverse position along the electron trajectory.
        
    u_z : array 
        electron longitudinal speed 

    omega : array
        radiation frequency
    
    order : int, optional
        ~1/gamma_z**2 function integration order. The default is 3.
    
    gradient_term : boolean, optional
        Account for the gradient term (imporstant in THz for accounting eadge radiation). The default is False.
    
    Green_func_type : TYPE, optional
        Type of the Green fuction to be untegrated, may be 'free_space', 'iris', 'pipe'. The default is 'free_space'.

    Returns
    -------
    Integrand exprestion f(.) that will be integrated to obtain the electic field
    '''
    print('\n Calculating Green_func_integrand...')
    start = time.time()
    
    if Green_func_type=='free_space':
        Green = G_free_space(x0, y0, z0, x, y, z, omega)
    
    elif Green_func_type=='iris':
        Green = G_iris(x0, y0, z0, x, y, z, omega, **kwargs)
    
    elif Green_func_type=='pipe':
        Green = G_pipe(x0, y0, z0, x, y, z, omega)
        
    else:
        print('please check type of the Green function')
    
    Int_z = np.array(gamma_z_integral(u_z, z, omega, order=order))

    Gf_x = lambda ux, Green, Int_z, omega: (4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*ux[:, np.newaxis, np.newaxis, np.newaxis]*Green) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
    Gf_y = lambda uy, Green, Int_z, omega: (4*np.pi*e_/speed_of_light) * ((1j*omega/speed_of_light**2)*uy[:, np.newaxis, np.newaxis, np.newaxis]*Green) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])

    if gradient_term==True:
        nub_G_x, nub_G_y = nubla_G(Green, x0, y0, z0, x, y, z, Green_func_type=Green_func_type, **kwargs)

        nub_G_x, nub_G_y = nub_G_x[:-1, :, :, :], nub_G_y[:-1, :, :, :]
        Green = Green[:-1, :, :, :]

        nubla_Gf_x = lambda nub_G_x, Int_z: (4*np.pi*e_/speed_of_light) * nub_G_x * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
        nubla_Gf_y = lambda nub_G_y, Int_z: (4*np.pi*e_/speed_of_light) * nub_G_y * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
        
        t_func = time.time() - start
        print(' Green_func_integrand is calculated in %.2f ' % t_func + 'sec \n')
        return Gf_x(ux, Green, Int_z, omega) + nubla_Gf_x(nub_G_x, Int_z), Gf_y(uy, Green, Int_z, omega) + nubla_Gf_y(nub_G_y, Int_z)
        # return nubla_Gf_x(nub_G_x, Int_z), nubla_Gf_y(nub_G_y, Int_z),
        # return nubla_Gf(ux, Green, Int_z, omega)      
    else:
        Green = Green[:-1, :, :, :]
        t_func = time.time() - start
        print(' Green_func_integrand is calculated in %.2f ' % t_func + 'sec \n')
        return Gf_x(ux, Green, Int_z, omega), Gf_y(ux, Green, Int_z, omega)
     
def Green_func_integarator(f, z, order=3):
    '''
    Parameters
    ----------
    f : array
        Integrand
    order : int, optional
        Integration order. The default is 3.

    Returns
    -------
    The field E_x, E_y
    '''
    print('\n Calculating radition field...')
    start = time.time()
    
    z_dz = np.roll(z, -1)[:-1]
    dz = np.array(z_dz - z[:-1])
    fx = f[0]
    fy = f[1]
    
    if order==2:
        E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis] * (fx), axis=0) #oder 2
        E_y = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis] * (fy), axis=0) #oder 2
    elif order==3:
        E_x = np.sum(dz[:-1, np.newaxis, np.newaxis, np.newaxis] * (fx[:-1] + np.roll(fx, -1,  axis=0)[:-1])/2, axis=0) #oder 3
        E_y = np.sum(dz[:-1, np.newaxis, np.newaxis, np.newaxis] * (fy[:-1] + np.roll(fy, -1,  axis=0)[:-1])/2, axis=0) #oder 3

    elif order==4:
        # TO BE DONE
        # E_x = np.sum((f[:-1] + 4* + np.roll(f, -1,  axis=0)[:-1]) * dz[:, np.newaxis, np.newaxis, np.newaxis] / 6, axis=0) #oder 4 to be done
        return True  

    t_func = time.time() - start
    print(' Radition field is calculated in %.2f ' % t_func + 'sec \n')    

    return E_x, E_y

def SR_field2Intensity(E_x, E_y, out="Ix", norm='e_beam'):
    '''
    Parameters
    ----------
    Ex : TYPE
        DESCRIPTION.
    Ey : TYPE
        DESCRIPTION.
    out : TYPE, optional
        DESCRIPTION. The default is "Ix".
    norm : TYPE, optional
        DESCRIPTION. The default is 'e_beam'.

    Returns
    -------
    None.

    '''
    h_erg_s = 1.054571817 * 1e-27 
    h_J_s = 1.054571817 * 1e-34 

    speed_of_light_SGS = speed_of_light * 1e2
    A = 0.4 * speed_of_light / (e_ * h_erg_s * 4 * np.pi**2) / (1e7/speed_of_light**2) * 1e-2#* SI unit 1e-4 is due to cm^2 -> mm^2, (1e7/speed_of_light**2) is the factor due to transformation of (I/eh_bar) * c * E**2, I -> I * 1e-1 c, e -> e * 1e-1 c, h_bar -> h_bar * 1e7, c -> c * 1e2, E -> E * 1e6 / c 

    I = A * (np.real(E_x)**2 + np.imag(E_x)**2 + np.real(E_y)**2 + np.imag(E_y)**2)
    return I


#%%
beam = Beam()
beam.E = 4.0            # beam energy in [GeV]
beam.I = 0.4 
gamma = beam.E/0.5109906e-03 #relative energy E_electron/m_e [GeV/Gev]
beta = np.sqrt(1 - 1/gamma**2)
e_ = 1.60218e-19 #elementary charge

# B0 = 0.4
# lperiod = 0.03# [m] undulator period 

B0 = 1
lperiod = 0.5# [m] undulator period 

K = 0.9336 * B0 * lperiod * 100 # longitudinal coordinates from 0 to lperiod*nperiods in [mm] 
# und = Undulator(Kx=K, nperiods=1.5, lperiod=lperiod, eid="und", phase=0, end_poles=1)
und = Undulator(Kx=K, nperiods=0.5, lperiod=lperiod, eid="und", phase=0, end_poles=0)
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
U, E = track4rad_beam(p_array, lat, energy_loss=False, quantum_diff=False, accuracy=1)#
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


z0 = 2
# xy=5e-4
xy=0

screen = Screen()
screen.z = z0     # distance from the begining of lattice to the screen 
screen.start_energy = THz*(1 - 0.9999)*1e12*h_eV_s #0.000001#1e12*h_eV_s    # [eV], starting photon energy
screen.end_energy =   THz*(1 + 4)*1e12*h_eV_s #0.12#10e12*h_eV_s      # [eV], ending photon energy
screen.num_energy = 300
screen.x = -xy # half of screen size in [m] in horizontal plane
screen.y = -xy   # half of screen size in [m] in vertical plane
screen.nx = 1       # number of poinfts in horizontal plane 
screen.ny = 1    # number of points in vertical plane 

screen = calculate_radiation(lat, screen, beam, energy_loss=False, quantum_diff=False, accuracy=1)
data = screen.Total#/max(screen.Total)
X = screen.Eph


ux = xp[:-1]*beta*speed_of_light
uy = yp[:-1]*beta*speed_of_light
u_z = np.sqrt((speed_of_light*beta)**2 - ux**2 - uy**2)

omega_r = 2 * np.pi * speed_of_light / lambds
omega = np.linspace(omega_r*(1-0.9999), omega_r*(1+4), 300)

# x0 = np.array([0]) #np.linspace(-xy, xy, 1)
# y0 = np.array([0]) #np.linspace(-xy, xy, 1)
x0 = np.linspace(-xy, xy, 1)
y0 = np.linspace(-xy, xy, 1)

n_list = [0]
k_0 = 1
k_list=np.arange(k_0, k_0 + 2, 1)


iris={'a' : 0.02, 'b' : 0.01, 'n_list' : n_list, 'k_list' : k_list}
f = Green_func_integrand(x0, y0, z0, x, y, z, omega, ux, uy, u_z, order=3, gradient_term=True, Green_func_type='iris', **iris)
E_x, E_y = Green_func_integarator(f, z, order=3)
I = SR_field2Intensity(E_x, 0, out="Ix", norm='e_beam')[0,0,:]


def E_ph2THz(x):
    return x * h_eV_s / 1e-12

def THz2E_ph(x):
    return x / h_eV_s*1e-12

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(X/h_eV_s*1e-12, data, linewidth=1, color='black', linestyle='-', label='OCELOT')
ax1.plot(omega*1e-12/2/np.pi, I, label=r'Green func', linestyle='-.', linewidth=1, color='blue', alpha=1)

secax = ax1.secondary_xaxis('top', functions=(E_ph2THz, THz2E_ph))
secax.set_xlabel(r'$E_{ph}$, [eV]', fontsize=14)

ax1.set_yscale('log')
ax2 = ax1.twinx() 
ax2.plot(X/h_eV_s*1e-12, data, linewidth=1, color='black', linestyle='-', label='OCELOT')
ax2.plot(omega*1e-12/2/np.pi, I, label=r'Green func', linestyle='-.', linewidth=1, color='blue', alpha=1)
ax2.set_yticks([])
# ax2.set_yscale('log')


ax3 = ax1.twinx() 
ax3.plot(omega*1e-12/2/np.pi, np.where(abs((data - I)/(data))*1e2<=10, (data - I)/(data)*1e2, None), label=r'Norm. discrepancy', linewidth=0.75, color='red', alpha=1)
# ax3.plot(omega*1e-12/2/np.pi, (data - I)/(data)*1e2, label=r'Norm. discrepancy', linewidth=0.5, color='red', alpha=1)

# ax3.set_yscale('log')
ax3.set_ylabel('($I_{Green}$ - $I_{oclot}$)/$I_{Green}$, $\%$', fontsize=14, color='red')
ax3.tick_params(axis='y', colors='red') 
secax3 = ax2.secondary_yaxis(-0.13)
secax3.set_ylabel('Flux, arb.units', fontsize=14)


ax1.set_ylim(1e4)
ax1.set_xlim(0, np.max(omega*1e-12/2/np.pi))
ax2.set_ylim(0)
ax2.set_xlim(0, np.max(omega*1e-12/2/np.pi))

ax1.legend(loc=2)
ax3.legend(loc=1)

ax1.set_xlabel(r'$\Omega$, [THz]', fontsize=14)
# ax1.set_xlabel('E_ph, eV', fontsize=14)
ax1.grid()
plt.tight_layout()
plt.show()

#%%
# THz = 24.5
THz =  speed_of_light * 1e-12 / lambds#0.000299792458/lambds

omega = np.array([THz*1e12 * 2* np.pi])

print(E_ph2THz(omega))

# xy = 0.075
xy = 0.02
n = 100
x0 = np.linspace(-xy, xy, n)
y0 = np.linspace(-xy, xy, n)
z0 = 2

f = Green_func_integrand(x0, y0, z0, x, y, z, omega, ux, uy, u_z, order=3, gradient_term=True, Green_func_type='iris', **iris)
E_x, E_y = Green_func_integarator(f, z, order=3) 
I = SR_field2Intensity(E_x, 0, out="Ix", norm='e_beam')

screen = Screen()
screen.z = z0      # distance from the begining of lattice to the screen 
screen.size_x = xy # half of screen size in [m] in horizontal plane
screen.size_y = xy   # half of screen size in [m] in vertical plane
screen.nx = n       # number of poinfts in horizontal plane 
screen.ny = n    # number of points in vertical plane 

# X/h_eV_s*1e-12
screen.start_energy = THz*1e12*h_eV_s# [eV], starting photon energy
screen.end_energy =   THz*1e12*h_eV_s     # [eV], ending photon energy
screen.num_energy = 1  
screen = calculate_radiation(lat, screen, beam, energy_loss=False, quantum_diff=False, accuracy=1)

data = screen.Total

dfl = RadiationField()
dfl = screen2dfl(screen, polarization='x')
constQuant = 3*alpha/(q_e*1e2)/(4*pi**2)*1e-3 * beam.I * gamma * gamma / (z0*1e2)**2
I_oclot= dfl.intensity() * constQuant

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, num='2D', figsize=(15, 5))
fig.suptitle("{} THz, {} eV, z0 = {} m".format(round(THz), round(THz*1e12*h_eV_s, 2), z0), fontsize=16)

ax1.pcolormesh(dfl.scale_x(), dfl.scale_y(), I_oclot[0, :, :])
ax1.set_title("OCELOT")
ax2.pcolormesh(x0, y0, I[:, :, 0])
ax2.set_title("Green func")
ax2.set_yticks([])

im = ax3.pcolormesh(x0, y0, (I[:, :, 0] - I_oclot[0, :, :])/np.max(I) *1e2, cmap='seismic')
ax3.set_title("Discrepancy")
cax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
fig.colorbar(im, cax=cax, use_gridspec=0)
cax.set_ylabel('($I_{Green}$ - $I_{oclot}$)/max($I_{Green}$), $\%$', fontsize=16)
ax3.set_yticks([])


ax1.set_xlabel('x, [m]', fontsize=14)
ax2.set_xlabel('x, [m]', fontsize=14)
ax1.set_ylabel('y, [m]', fontsize=14)
# ax2.set_ylabel('y, [m]')
plt.show()
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, num='1D', figsize=(10, 5),)
fig.suptitle("Slices", fontsize=16)

ax1.plot(dfl.scale_x(), I_oclot[0, n//2, :], c='black', label="OCELOT")
ax1.plot(x0, I[n//2, :, 0], '-.', c='blue', label="Green func")
ax1.set_ylim(0)

ax1_twin = ax1.twinx() 
ax1_twin.plot(y0, (I[n//2, :, 0] - I_oclot[0, n//2, :])/np.max(I) *1e2, label=r'Norm. discrepancy', linewidth=0.75, color='red', alpha=1)
ax1_twin.set_ylabel('Discrepancy, $\%$', fontsize=16, c='red')
ax1_twin.tick_params(axis='y', colors='red') 

# ax1_twin.set_yscale('log')

ax2.plot(dfl.scale_y(), I_oclot[0, :, n//2], c='black', label="OCELOT")
ax2.plot(y0, I[:, n//2, 0], '-.', c='blue', label="Green func")
ax2.set_ylim(0)

ax2_twin = ax2.twinx() 
ax2_twin.plot(y0, (I[:, n//2, 0] - I_oclot[0, :, n//2])/np.max(I) *1e2, label=r'Norm. discrepancy', linewidth=0.75, color='red', alpha=1)
ax2_twin.set_ylabel('Discrepancy, $\%$', fontsize=16, c='red')
ax2_twin.tick_params(axis='y', colors='red') 

# ax2_twin.set_yscale('log')


ax1.set_xlabel('x, [m]', fontsize=14)
ax2.set_xlabel('y, [m]', fontsize=14)
ax1.set_ylabel('Flux', fontsize=14)
ax2.set_ylabel('Flux', fontsize=14)


ax1.legend()
ax1_twin.legend()
plt.tight_layout()
plt.show()
