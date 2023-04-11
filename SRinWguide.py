import numpy as np
import matplotlib as plt
from matplotlib import colors

from scipy.integrate import quad
from scipy import special
import itertools
from operator import add
import time
from tqdm import tqdm

from ocelot.gui import *
from ocelot.common.globals import *  # import of constants like "h_eV_s" and
from ocelot.cpbd.elements import Undulator
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.beam import *
from ocelot.rad.screen import Screen
from ocelot.rad.radiation_py import *
from ocelot.optics.wave import dfl_waistscan, screen2dfl, RadiationField
from ocelot.gui.dfl_plot import plot_dfl, plot_dfl_waistscan


#the dynamic memory allocation version

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
    array with the shape [Nz, Nx, Nx, omega]
        Green function
    '''
    print('    calculating G_free_space...')
    start = time.time()
    x0, y0 = np.meshgrid(x0, y0)
    return (-1/(4*np.pi*(z0 - z[:, np.newaxis, np.newaxis, np.newaxis]))) * np.exp(1j*omega[np.newaxis, np.newaxis, np.newaxis, :] * ((x0[np.newaxis, :, :, np.newaxis] - x[:, np.newaxis, np.newaxis, np.newaxis])**2 + (y0[np.newaxis, :, :, np.newaxis] - y[:, np.newaxis, np.newaxis, np.newaxis])**2)/(2 * speed_of_light * (z0 - z[:, np.newaxis, np.newaxis, np.newaxis])))

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
    array with the shape [Nz, Nx, Nx, omega]
        iris stracture Green function
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
    for nk in tqdm(list(itertools.product(n_list, k_list))):
        n = nk[0]
        k = nk[1]
        v_nk = special.jn_zeros(n, k)[-1]
        # print('    n =', n, 'k =', k, 'v_nk =', v_nk)
        
        G_nk = -((1j * speed_of_light * (1.0 - 2.0*delta[np.newaxis, np.newaxis, np.newaxis, :])) / (2 * np.pi * omega[np.newaxis, np.newaxis, np.newaxis, :] * a**2)) * \
            (np.exp(-1j*n*(phi0[np.newaxis, :, :, np.newaxis] - phi[:, np.newaxis, np.newaxis, np.newaxis]))) / (special.jv(n+1, v_nk))**2 * \
                np.exp(-(1j*speed_of_light*(z0 - z[:,np.newaxis,np.newaxis,np.newaxis]) * v_nk**2 * (1.0 - 2.0*delta[np.newaxis,np.newaxis,np.newaxis,:])) / (2 * omega[np.newaxis,np.newaxis,np.newaxis,:] * a**2)) * \
                    special.jv(n, v_nk * (1 - delta[np.newaxis,np.newaxis,np.newaxis,:])*r[:, np.newaxis, np.newaxis, np.newaxis]/a) * \
                        special.jv(n, v_nk * (1 - delta[np.newaxis,np.newaxis,np.newaxis,:])*r0[np.newaxis, :, :, np.newaxis]/a)
        
        Green = np.add(Green, G_nk)  #still summing problem here! or not!? 

    t_func = time.time() - start
    print('        calculated G_iris in %.2f ' % t_func + 'sec')
        
    return np.array(Green)

def G_pipe(x0, y0, z0, x, y, z, omega, R = 0.02, m_list=[0], k_list=[1]):
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
    array with the shape [2, 2, Nz, Nx, Nx, omega]
        pipe Green function
    '''
    print('    calculating G_pipe...')
    start = time.time()
    print('        R =', R, 'n_list =', m_list, 'k_list =', k_list)

    x0, y0 = np.meshgrid(x0, y0)
    
    phi0 = np.arctan2(y0, x0)
    phi = np.arctan2(y, x)
    
    r0 = np.sqrt(x0**2 + y0**2)
    r =  np.sqrt(x**2 + y**2)
    
    Green = np.zeros((np.shape(z)[0], np.shape(x0)[0], np.shape(y0)[0], np.shape(omega)[0]))
    for mk in tqdm(list(itertools.product(m_list, k_list))):
        m = mk[0]
        k = mk[1]
        
        mu_mk = special.jnp_zeros(m, k)[-1]
        nu_mk =  special.jn_zeros(m, k)[-1]
        
        # print('shape mu, nu = ', mu_mk, nu_mk, np.shape(mu_mk), np.shape(nu_mk))
        a_m=0
        if m == 0:
            a_m = 1
        elif m >= 1:
            a_m = 2

        A_TE_mk = np.sqrt(a_m/np.pi)/(special.jv(m, mu_mk)*np.sqrt(mu_mk**2 - m**2))
        A_TM_mk = np.sqrt(a_m/np.pi)/(nu_mk*special.jv(m-1, nu_mk))
        
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
       
        # print(np.shape(M1_TE), np.shape(M2_TE), np.shape(M1_TM), np.shape(M2_TM))
        # print(np.shape(g_TE), np.shape(g_TM))
        G_nk = g_TE[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :] * (M1_TE[:, :, :, :, :,np.newaxis] + M2_TE[:, :, :, :, :,np.newaxis]) + g_TM[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :] * (M1_TM[:, :, :, :, :,np.newaxis] + M2_TM[:, :, :, :, :,np.newaxis])
        # print(G_nk)
        # G_12 = g_TE[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TE[1,0, :, :, :,np.newaxis] + M2_TE[1,0, :, :, :,np.newaxis]) + g_TM[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TM[1,0, :, :, :,np.newaxis] + M2_TM[1,0, :, :, :,np.newaxis])
        # G_22 = g_TE[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TE[1,1, :, :, :,np.newaxis] + M2_TE[1,1, :, :, :,np.newaxis]) + g_TM[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TM[1,1, :, :, :,np.newaxis] + M2_TM[1,1, :, :, :,np.newaxis])
        # G_21 = g_TE[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TE[0,1, :, :, :,np.newaxis] + M2_TE[0,1, :, :, :,np.newaxis]) + g_TM[np.newaxis, np.newaxis, np.newaxis, :] * (M1_TM[0,1, :, :, :,np.newaxis] + M2_TM[0,1, :, :, :,np.newaxis])
        
        # Green_11 = G_11 + Green_11
        # Green_12 = G_12 + Green_12
        # Green_22 = G_22 + Green_22
        # Green_21 = G_21 + Green_21
        Green = np.add(G_nk, Green)  #still summing problem here! or not!?  
        # Green = G_nk + Green

        # print('            Green function shape = ', np.shape(Green))
        t_func = time.time() - start
    
    print('    calculated G_pipe in %.2f ' % t_func + 'sec')

    return Green


def nubla_G(Green, x0, y0, z0, x, y, z, omega, h='machine', Green_func_type='free_space', **kwargs):
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
    array with shape [Nz, Nx, Nx, omega] for free space and iris structure or [2, 2, Nz, Nx, Nx, omega] for pipe
        gradient components of the given Green function

    '''
    print('    calculating nubla_G...')
    start = time.time()

    epsilon=1.0
    while epsilon+1>1:
        epsilon=epsilon/2
    epsilon=epsilon*1e1
    print("The value of epsilon is: ", epsilon)
    
    if h=='machine':
        # bool_x = np.where(x!=0, True, False)
        dx = np.where(x!=0, np.sqrt(epsilon)*abs(x), np.sqrt(epsilon)*1)
        dy = np.where(y!=0, np.sqrt(epsilon)*abs(y), np.sqrt(epsilon)*1)
        # print("The value of dx and dy is: ", dx, dx.shape)

    elif isinstance(dx, float):
        dx = h
        dy = h
    else: 
        ValueError('please, enter a float "h" for taking derivative')
    
    if Green_func_type == 'free_space':
        Green  = G_free_space(x0, y0, z0, x, y, z, omega)        
        dG_x = G_free_space(x0, y0, z0, x + dx, y, z, omega) - Green
        dG_y = G_free_space(x0, y0, z0, x, y + dy, z, omega) - Green
        
        return dG_x/dx[:, np.newaxis, np.newaxis, np.newaxis], dG_y/dy[:, np.newaxis, np.newaxis, np.newaxis]

    elif Green_func_type == 'iris':
        dG_x = G_iris(x0, y0, z0, x + dx, y, z, omega, **kwargs) - Green
        dG_y = G_iris(x0, y0, z0, x, y + dy, z, omega, **kwargs) - Green
        return dG_x/dx[:, np.newaxis, np.newaxis, np.newaxis], dG_y/dy[:, np.newaxis, np.newaxis, np.newaxis]

    elif Green_func_type == 'pipe':
        dG_x = G_pipe(x0, y0, z0, x + dx, y, z, omega, **kwargs) - Green
        dG_y = G_pipe(x0, y0, z0, x, y + dy, z, omega, **kwargs) - Green
        return dG_x/dx[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis], dG_y/dy[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]

    else:
        print('plese enter right "Green_func_type": "free_space", "iris", "pipe"')
        
    t_func = time.time() - start
    print('        calculated nubla_G in %.2f ' % t_func + 'sec')

def gamma_z(u_z):
    '''
    Parameters
    ----------
    u_z : float
        Longitudinal speed.

    Returns
    -------
    Float with z integration size.

    '''
    return 1/np.sqrt(1 - u_z**2/speed_of_light**2)

f_gamma = lambda u_z, omega: np.array((omega[np.newaxis, :]/2/speed_of_light)*(1/gamma_z(u_z)[:, np.newaxis]**2))
def gamma_z_integral(u_z, z, omega, order=3):
    '''
    Parameters
    ----------
    u_z : float
        Longitudinal speed.
    z : array
        integration position along the electron trajectory.
    omega : array
        Given frequency.
    order : int, optional
        Order of the integration according to Runge-Kutta methods. The default is 3.

    Returns
    -------
    I_array : array 
    '''
    print('\n    taking gamma_z_integral...')
    start = time.time()
    # z_dz = np.roll(z, -1)[:-1]
    # dz = np.array(z_dz - z[:-1])
    
    I_array = np.array([])
    func = f_gamma(u_z, omega)
    
    print('\n      integration order is ', order)
    dz = z[1:] - z[:-1]

    if order==2:
        I_array = np.array([np.sum(func[0:i]*dz[0:i, np.newaxis], axis=0) for i in range(0, np.shape(z)[0]-1)])
        I_array = np.append(I_array, I_array[-1, :][np.newaxis, :] + func[-1, :][np.newaxis, :]*dz[-1], axis=0)

    elif order==3:
        I_array = np.array([np.sum((func[0:-1, :][:i, :] + func[1:, :][:i, :])*dz[0:i, np.newaxis]/2, axis=0) for i in range(0, np.shape(z)[0]-1)])
        I_array = np.append(I_array, I_array[-1, :][np.newaxis, :] + func[-1, :][np.newaxis, :]*dz[-1], axis=0)
        
    elif order==4:
        u_z_dz = np.roll(u_z, -1)[:-1]
        func_dfunc = f_gamma((u_z[:-1] + u_z_dz)/2, omega)
        I_array = np.array([np.sum((func[0:i] + 4*func_dfunc[0:i] + np.roll(func, -1, axis=0)[0:i]) * dz[0:i, np.newaxis]/6, axis=0) for i in range(np.shape(z)[0]-1)])
    
    elif order==5:
        size = len(z)
        Nmotion = int((size + 1)/3)
        a, b = z[0], z[-1]
        half_step = (b - a) / 2. / (Nmotion - 1)
        w = [5/9, 8/9, 5/9]
        func_gaus = func[1:-1] * np.array(int(size/3)*w)[:, np.newaxis]
        
        # print("func_gaus", func_gaus.shape)
        I_array = np.array([np.sum((func_gaus[0:-3:3]+func_gaus[1:-2:3]+func_gaus[2:-1:3])[:i], axis=0) for i in range(Nmotion-1)]) * half_step
        I_array = np.append(I_array, I_array[-1, :][np.newaxis, :] + 2*func[-1, :][np.newaxis, :]*half_step, axis=0) #questionable

    else:
        print('the integration order might be O(h^2), O(h^3), O(h^4), please, enter 2,3,4 correspondingly')
    
    t_func = time.time() - start
    print('    tgamma_z_integral is taken in %.2f ' % t_func + 'sec')

    return I_array

def cor2gaus_cor(x, y, z, ux, uy, uz):
    z_gaus = x2xgaus(z)
    x_gaus = bspline(z, x, z_gaus)
    y_gaus = bspline(z, y, z_gaus)
    ux_gaus = bspline(z, ux, z_gaus)
    uy_gaus = bspline(z, uy, z_gaus)
    uz_gaus = bspline(z, uz, z_gaus)
    
    return x_gaus, y_gaus, z_gaus, ux_gaus, uy_gaus, uz_gaus

def Green_func_integrand(x0, y0, z0, x, y, z, omega, ux, uy, uz, order=5, gradient_term=True, Green_func_type='free_space', **kwargs):
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
        Account for the gradient term (important in THz for accounting edge radiation). The default is False.
    
    Green_func_type : TYPE, optional
        Type of the Green function to be integrated can be 'free_space', 'iris', or 'pipe'. The default is 'free_space'.

    Returns
    -------
    Integrand exprestion f(.) that will be integrated to obtain the electic field
    '''
    print('\n Calculating Green_func_integrand...')
    start = time.time()
    
    if order==5:
        z_old = z
        x, y, z, ux, uy, uz = cor2gaus_cor(x, y, z, ux, uy, uz) 
        Int_z = gamma_z_integral(uz, z, omega, order=order)
        Int_z = np.array([bspline(z_old, Int_z[:, i], z) for i in range(0, np.shape(Int_z)[1])]).T
        del z_old
    else:
        Int_z = gamma_z_integral(uz, z, omega, order=order)
    
    print('    Green_func_type is', Green_func_type)
    if Green_func_type=='free_space':
        Green = G_free_space(x0, y0, z0, x, y, z, omega)
    
    elif Green_func_type=='iris':
        Green = G_iris(x0, y0, z0, x, y, z, omega, **kwargs)
    
    elif Green_func_type=='pipe':
        Green = G_pipe(x0, y0, z0, x, y, z, omega, **kwargs)
        
    else:
        print('please check type of the Green function')
    
    #current term
    if Green_func_type=='free_space' or Green_func_type=='iris':
        # Green = Green[:-1, :, :, :]
        # ux = ux[:-1]
        # uy = uy[:-1]
        print(np.shape(Green), np.shape(Int_z), np.shape(ux))

        Gf_x = lambda ux, Green, Int_z, omega: (4*np.pi*q_e/speed_of_light) * ((1j*omega/speed_of_light**2)*ux[:, np.newaxis, np.newaxis, np.newaxis]*Green) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
        Gf_y = lambda uy, Green, Int_z, omega: (4*np.pi*q_e/speed_of_light) * ((1j*omega/speed_of_light**2)*uy[:, np.newaxis, np.newaxis, np.newaxis]*Green) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
    
    elif Green_func_type=='pipe':
        Gf_x = lambda ux, Green, Int_z, omega: (4*np.pi*q_e/speed_of_light) * ((1j*omega/speed_of_light**2)*(ux[:, np.newaxis, np.newaxis, np.newaxis]*Green[0,0,:,:,:,:] +\
                                                                                                            uy[:, np.newaxis, np.newaxis, np.newaxis]*Green[0,1,:,:,:,:])) * \
                                                                                                                                np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])

        Gf_y = lambda ux, Green, Int_z, omega: (4*np.pi*q_e/speed_of_light) * ((1j*omega/speed_of_light**2)*(ux[:, np.newaxis, np.newaxis, np.newaxis]*Green[1,0,:,:,:,:] +\
                                                                                                            uy[:, np.newaxis, np.newaxis, np.newaxis]*Green[1,1,:,:,:,:])) * \
                                                                                                                                np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
    else: 
        print('enter right type of the Green function: "free_space", "iris" or "pipe"')                                                                                                                
    
    #gradient term
    if gradient_term==True:
        nub_G_x, nub_G_y = nubla_G(Green, x0, y0, z0, x, y, z, omega, Green_func_type=Green_func_type, **kwargs)

        if Green_func_type=='free_space' or Green_func_type=='iris':
            # print(np.shape(Green), np.shape(Int_z), np.shape(ux), np.shape(nub_G_x))

            # Green = Green[:-1, :, :, :]
            # ux = ux[:-1]

            nubla_Gf_x = lambda nub_G_x, Int_z: (4*np.pi*q_e/speed_of_light) * nub_G_x * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
            nubla_Gf_y = lambda nub_G_y, Int_z: (4*np.pi*q_e/speed_of_light) * nub_G_y * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
        
        elif Green_func_type=='pipe':
            # nub_G_x, nub_G_y = nub_G_x[:, :, :-1, :, :, :], nub_G_y[:, :, :-1, :, :, :] 
            # Green = Green[:, :, :-1, :, :, :]
            
            nubla_Gf_x = lambda nub_G_x, Int_z: (4*np.pi*q_e/speed_of_light) * (nub_G_x[0,0,:,:,:,:] + nub_G_y[0,1,:,:,:,:]) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
            nubla_Gf_y = lambda nub_G_y, Int_z: (4*np.pi*q_e/speed_of_light) * (nub_G_x[1,0,:,:,:,:] + nub_G_y[1,1,:,:,:,:]) * np.exp(1j*Int_z[:, np.newaxis, np.newaxis, :])
        
        else: 
            print('enter right type of the Green function: "free_space", "iris" or "pipe"')            
        
        t_func = time.time() - start
        print(' Green_func_integrand is calculated in %.2f ' % t_func + 'sec \n')
        
        
        return Gf_x(ux, Green, Int_z, omega) + nubla_Gf_x(nub_G_x, Int_z), Gf_y(uy, Green, Int_z, omega) + nubla_Gf_y(nub_G_y, Int_z)
        # return nubla_Gf_x(nub_G_x, Int_z), nubla_Gf_y(nub_G_y, Int_z)

        # return nubla_Gf_x(nub_G_x, Int_z), nubla_Gf_y(nub_G_y, Int_z),
        # return Gf_x(ux, Green, Int_z, omega), nubla_Gf_x(nub_G_x, Int_z)

            # return nubla_Gf(ux, Green, Int_z, omega)      
    else:

        t_func = time.time() - start
        print(' Green_func_integrand is calculated in %.2f ' % t_func + 'sec')

        return Gf_x(ux, Green, Int_z, omega), Gf_y(uy, Green, Int_z, omega)
     
def Green_func_integrator(f, z, order=3):
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
    
    # z_dz = np.roll(z, -1)[:-1]
    # dz = np.array(z_dz - z[:-1])
    
    fx = f[0]
    fy = f[1]
    
    E_x = np.zeros((np.shape(fx)[0], np.shape(fx)[1], np.shape(fx)[2]))
    E_y = np.zeros((np.shape(fy)[0], np.shape(fy)[1], np.shape(fy)[2]))

    dz = z[1:] - z[:-1]

    if order==2:#oder 2
        E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis] * (fx[:-1]), axis=0)
        E_y = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis] * (fy[:-1]), axis=0)
    
    elif order==3:
        E_x = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis] * (fx[0:-1] + fx[1:])/2, axis=0) #oder 3
        E_y = np.sum(dz[:, np.newaxis, np.newaxis, np.newaxis] * (fy[0:-1] + fy[1:])/2, axis=0) #oder 3
        
        # E_x = np.sum(dz[:-1, np.newaxis, np.newaxis, np.newaxis] * (fx[:-1] + np.roll(fx, -1,  axis=0)[:-1])/2, axis=0) #oder 3
        # E_y = np.sum(dz[:-1, np.newaxis, np.newaxis, np.newaxis] * (fy[:-1] + np.roll(fy, -1,  axis=0)[:-1])/2, axis=0) #oder 3
    
    elif order==4:
        # E_x = np.sum((f[:-1] + 4* + np.roll(f, -1,  axis=0)[:-1]) * dz[:, np.newaxis, np.newaxis, np.newaxis] / 6, axis=0) #oder 4 to be done
        print('4th order has not been implemented…')
        return True  
    
    elif order==5:
        z = x2xgaus(z)
        size = len(z)
        Nmotion = int((size + 1)/3)
        a, b = z[0], z[-1]

        half_step = (b - a) / 2. / (Nmotion - 1)

        w = [5/9, 8/9, 5/9]

        fx_gaus = fx[1:-1] * np.array(int(size/3)*w)[:, np.newaxis, np.newaxis, np.newaxis]
        fy_gaus = fy[1:-1] * np.array(int(size/3)*w)[:, np.newaxis, np.newaxis, np.newaxis]

        E_x = np.array(np.sum((fx_gaus[0:-3:3, :, :, :] + fx_gaus[1:-2:3, :, :, :] + fx_gaus[2:-1:3, :, :, :]), axis=0))*half_step
        E_y = np.array(np.sum((fy_gaus[0:-3:3, :, :, :] + fy_gaus[1:-2:3, :, :, :] + fy_gaus[2:-1:3, :, :, :]), axis=0))*half_step

    t_func = time.time() - start
    print(' Radition field is calculated in %.2f ' % t_func + 'sec \n')    

    return E_x, E_y

def SR_field2Intensity(E_x, E_y, I_ebeam=0, out="Components", norm='e_beam'):
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
    
    Ix = A * (np.real(E_x)**2 + np.imag(E_x)**2) 
    Iy = A * (np.real(E_y)**2 + np.imag(E_y)**2)
    
    if out=='Components':
        return Ix, Iy
    elif out=='Total':
        return Ix + Iy
    

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
        
    
    
    
    
