"""
Consider the macroscopic formulation of Goldstein-Taylor model in discrete velocity kinetic theory, i.e.
∂_t rho + ∂_x j = 0
∂_t j + (c^2)/(eps^2)∂_x rho = - (sigma)/(eps^2)j
with x ∈ R, t >= 0, rho = rho(x, t) total particles density, j = j(x, t) scaled flux, 
eps scaling parameter of the kinetic dynamics and sigma scattering coefficient
"""

import numpy as np

def solve_E_I(x_disc, t_disc, rho_0, j_0, eps = 1e-4, sigma = 4, c = 1):
    """
    Function that solves the Goldstein-Taylor model using the Euler-IMEX scheme up to time t_end with periodic boundary conditions.
    The scheme is Asymptotic Preserving (AP) w.r.t. the parameter eps.
    Input: x_disc --> the discretization of the space domain (np.array)
           t_disc --> the discretization of the time domain, the solver will compute the solutions for t ∈ [0, t_end]
           rho_0, j_0 --> initial conditions for rho and j, (np.array)
           eps --> scaling parameter of the kinetic dynamics
           sigma --> scattering coefficient
           c --> scaling coefficient of the flux j = c*(f_plus - f_minus)/eps
    Mind the fact that we must satisfy the CFL condition, i.e. Δt <= Δx^2       
    """

    delta_x = (x_disc[-1] - x_disc[0])/(len(x_disc)-1) # Δx
    #print(delta_x)
    delta_t = (t_disc[-1] - t_disc[0])/(len(t_disc)-1) # Δt
    #print(delta_t)
    rho_tot = np.empty((len(x_disc), len(t_disc)))
    rho_tot[:] = np.nan
    rho_tot[:, 0] = rho_0 # initial condition
    j_tot = np.empty((len(x_disc), len(t_disc)))
    j_tot[:] = np.nan
    j_tot[:, 0] = j_0 # initial condition

    c_1 = delta_t*(eps**2)/((eps**2)+sigma*delta_t)
    c_2 = ((c*delta_t)**2)/((eps**2)+sigma*delta_t)
    c_3 = delta_t*(c**2)/((eps**2)+sigma*delta_t)
    c_4 = sigma*delta_t/((eps**2)+sigma*delta_t)

    rho_c = rho_0 # current value
    j_c = j_0 # current value

    for n in range(len(t_disc)-1):
       rho_n = np.empty(len(x_disc))
       j_n = np.empty(len(x_disc))
       for m in range(len(x_disc)):
              #print(m)
              if (m != 0 and m != len(x_disc)-1): # not boundary points
                     #print("no bound")
                     rho_n[m] = rho_c[m] - c_1*(0.5*j_c[m+1] - 0.5*j_c[m-1])/delta_x + c_2*(rho_c[m-1] - 2*rho_c[m] + rho_c[m+1])/(delta_x**2)
                     j_n[m] = j_c[m] - c_3*(0.5*rho_c[m+1] - 0.5*rho_c[m-1])/delta_x - c_4*j_c[m]
              elif(m == 0): # left boundary point, periodic boundary condition enforced using the second last point
                     #print("zero")
                     rho_n[m] = rho_c[m] - c_1*(0.5*j_c[m+1] - 0.5*j_c[-2])/delta_x + c_2*(rho_c[-2] - 2*rho_c[m] + rho_c[m+1])/(delta_x**2)
                     j_n[m] = j_c[m] - c_3*(0.5*rho_c[m+1] - 0.5*rho_c[-2])/delta_x - c_4*j_c[m]
              else: # right boundary point, periodic boundary condition enforced using the second first point   
                     #print("end")
                     rho_n[m] = rho_c[m] - c_1*(0.5*j_c[1] - 0.5*j_c[m-1])/delta_x + c_2*(rho_c[m-1] - 2*rho_c[m] + rho_c[1])/(delta_x**2)
                     j_n[m] = j_c[m] - c_3*(0.5*rho_c[1] - 0.5*rho_c[m-1])/delta_x - c_4*j_c[m]
       
       rho_tot[:, n+1] = rho_n.copy() # save the solution for rho
       j_tot[:, n+1] = j_n.copy() # save the solution j
       rho_c = rho_n.copy() # update for next iteration
       j_c = j_n.copy()  # update for next iteration
       
    return rho_tot, j_tot


if __name__ == "__main__":
    from time import time
    m = 200
    n = 1500
    x_disc = np.linspace(-1,1,m+1)
    t_disc = np.linspace(0,0.1,n+1)
    print(len(x_disc))
    delta_x = (x_disc[-1] - x_disc[0])/(len(x_disc)-1)
    delta_t = (t_disc[-1] - t_disc[0])/(len(t_disc)-1)
    print(f"Δx = {delta_x}")
    print(f"Δt = {delta_t}")
    print(f"Δx**2 = {(delta_x)**2}")
    c = 1
    sigma = 4
    eps = 1e-4
    rho_0 = 6+3*np.cos(3*np.pi*x_disc)
    j_0 = (9*np.pi*(c**2)/sigma)*np.sin(3*np.pi*x_disc)
    #print(len(x_disc))
    rho_tot, j_tot = solve_E_I(x_disc, t_disc, rho_0, j_0, eps, sigma, c)
    import matplotlib.pyplot as plt
    X, T = np.meshgrid(x_disc, t_disc)
    fig=plt.figure()
    plt.contourf(T, X, np.transpose(j_tot), 30)
    plt.colorbar()
    plt.show()
    