#sci package
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#sympy
from sympy import *

#utility
from tqdm.notebook import tqdm

def inlet(Rd):
    Q = np.loadtxt("data/inflow.csv", delimiter=',')
    
    # time
    t = [ elem for elem in Q[:,0]]
    
    # Q to V
    q = [ 1e-6 * elem for elem in Q[:,1] ] # cm^3/s to m^3/s

    return interp1d(t, q, kind='linear', bounds_error=False, fill_value=q[0])

def AtoP(A, Rd, Pd, E, rho, h, mu):
    #Ad
    Ad = Rd * Rd * np.pi
    
    #beta
    beta = (4/3)*(np.sqrt(np.pi)*E*h)

    #Pd
    re = Pd + (beta / Ad) * (np.sqrt(A) - np.sqrt(Ad))

    assert Ad >= 0
    assert np.isnan(re) == False
    return re

#dU/dt + dF/dZ = S

def UtoF(U, Rd, Pd, E, rho, h, mu):
    #rho

    #pressure
    P = AtoP(A = U[0], Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu)
    F = np.zeros(2)
    F[0] = U[0] * U[1]
    F[1] = (0.5*U[1]*U[1]) + (P/rho)
    assert np.isnan(F[0]) == False 
    assert np.isnan(F[1]) == False
    return F

def UtoF_mesh(U_mesh, Rd, Pd, E, rho, h, mu):
    nx = U_mesh.shape[1]

    F_mesh = np.zeros((2,nx))
    for i in range(nx):
        U = U_mesh[:,i]
        F = UtoF(U = U, Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu)
        F_mesh[:,i] = F
    
    return F_mesh 

def UtoS(U, Rd, Pd, E, rho, h, mu):
    #rho
    
    #mu

    #f
    f = -8 * mu * np.pi * U[1] 
    
    S = np.zeros(2)
    S[1] = f / (rho * U[0])
    assert np.isnan(S[0]) == False 
    assert np.isnan(S[1]) == False
    return S

def UtoS_mesh(U_mesh, Rd, Pd, E, rho, h, mu):
    nx = U_mesh.shape[1]
    
    S_mesh = np.zeros((2,nx)) 
    for i in range(nx):
        U = U_mesh[:,i]
        S = UtoS(U, Rd, Pd, E, rho, h, mu)
        S_mesh[:,i] = S 
    
    return S_mesh

def QtoV(Q, A, Rd, Pd, E, rho, h, mu):
    re = Q / A
    assert np.isnan(re) == False
    return re

def VtoQ(V, A, Rd, Pd, E, rho, h, mu):
    re = V * A
    assert np.isnan(re) == False
    return re

#Rds, Es, Ls, hs, 
def get_residual_functions(dt, dx, \
    U_mesh_p, U_mesh_d1, U_mesh_d2, \
    U_mesh_half_p, U_mesh_half_d1, U_mesh_half_d2, \
    Rd_p, Rd_d1, Rd_d2, \
    E_p, E_d1, E_d2,  \
    h_p, h_d1, h_d2, \
    Pd, rho, mu
    ):

    #U F S 
    #U[0] : area
    #U[1] : average velocity 

    def AtoP(A, Rd, Pd, E, rho, h, mu):
        Ad = np.pi * Rd * Rd
        beta = (4/3)*(np.sqrt(np.pi)*E*h)
        re = Pd + (beta / Ad) * (np.sqrt(A) - np.sqrt(Ad))
        return P

    def F0(A, V, Rd, Pd, E, rho, h, mu):
        re = A * V
        return re

    def F1(A, V, Rd, Pd, E, rho, h, mu):
        P = AtoP(A_str, Rd, Pd, E, rho, h, mu)
        re = 0.5 * V * V + (1/rho) * P
        return re
    
    def S1(A_str, V_str, Rd, Pd, E, rho, h, mu):
        f = -8 * mu * np.pi * V
        re = f / (rho * A)
        return re

    U_prev_p = U_mesh_p[:,-1]
    U_mid_p = U_mesh_half_p[:,-1]

    U_prev_d1 = U_mesh_d1[:,0]
    U_mid_d1 = U_mesh_half_d1[:,0]
    
    U_prev_d2 = U_mesh_d2[:,0]
    U_mid_d2 = U_mesh_half_d2[:,0]

    #parent
    par_area = {}
    par_area['down'] = U_prev_p[0]
    par_area['up'] = Symbol('x9')
    par_area['mid'] = Symbol('x10')
    par_area['left'] = U_mid_p[0]
    par_area['right'] = Symbol('x11')
    par_vel = {}
    par_vel['down'] = U_prev_p[1]
    par_vel['up'] = Symbol('x0')
    par_vel['mid'] = Symbol('x1')
    par_vel['left'] = U_mid_p[1]
    par_vel['right'] = Symbol('x2')

    #d1
    d1_area = {}
    d1_area['down'] = U_prev_d1[0]
    d1_area['up'] = Symbol('x12')
    d1_area['mid'] = Symbol('x13')
    d1_area['left'] = Symbol('x14')
    d1_area['right'] = U_mid_d1[0]
    d1_vel = {}
    d1_vel['down'] = U_prev_d1[1]
    d1_vel['up'] = Symbol('x3')
    d1_vel['mid'] = Symbol('x4')
    d1_vel['left'] = Symbol('x5')
    d1_vel['right'] = U_mod_d1[1]

    #d2
    d2_area = {}
    d2_area['down'] = U_prev_d2[0]
    d2_area['up'] = Symbol('x15')
    d2_area['mid'] = Symbol('x16')
    d2_area['left'] = Symbol('x17')
    d2_area['right'] = U_mod_d2[0]
    d2_vel = {}
    d2_vel['down'] = U_prev_d2[1]
    d2_vel['up'] = Symbol('x6')
    d2_vel['mid'] = Symbol('x7')
    d2_vel['left'] = Symbol('x8')
    d2_vel['right'] = U_mod_d2[1]

    #f0 ~ f2 : navier second term
    #f0
    f0 = (Symbol(par_vel['up']) - par_vel['down'])
    + (dt / dx)
    (
        F1(A_str = par_area['right'], V_Str = par_vel['right'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
         - F1(A_str = par_area['left'], V_Str = par_vel['left'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
    ) 
    - (dt / 2)
    (
        S1(A_str = par_area['right'], V_Str = par_vel['right'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
         - S1(A_str = par_area['left'], V_Str = par_vel['left'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
    )
    #f1
    f1 = (Symbol(d1_vel['up']) - d1_vel['down']) 
    + (dt / dx)
    (
        F1(A_str = d1_area['right'], V_Str = d1_vel['right'], Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
         - F1(A_str = d1_area['left'], V_Str = d1_vel['left'], Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    ) 
    - (dt / 2)
    (
        S1(A_str = d1_area['right'], V_Str = par_vel['right'], Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
         - S1(A_str = d1_area['left'], V_Str = par_vel['left'], Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    )
    #f2
    f2 = (Symbol(d2_vel['up']) - d2_vel['down']) 
    + (dt / dx)
    (
        F1(A_str = d2_area['right'], V_Str = d2_vel['right'], Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
         - F1(A_str = d2_area['left'], V_Str = d2_vel['left'], Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
    ) 
    - (dt / 2)
    (
        S1(A_str = d2_area['right'], V_Str = d2_vel['right'], Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
         - S1(A_str = d2_area['left'], V_Str = d2_vel['left'], Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
    )
    #f3 ~ f5 : navier first term
    #f3
    f3 = (par_area['up'] - par_area['down'])
    + (dt / dx)
    (
        F0(A_str = par_area['right'], V_Str = par_area['right'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
        - F0(A_str = par_area['left'], V_Str = par_area['left'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
    )
    #f4
    f4 = (d1_area['up'] - d1_area['down'])
    + (dt / dx)
    (
        F0(A_str = d1_area['right'], V_Str = d1_area['right'], Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
        - F0(A_str = d1_area['left'], V_Str = d1_area['left'], Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    )   
    #f5
    f5 = (d2_area['up'] - d2_area['down'])
    + (dt / dx)
    (
        F0(A_str = d2_area['right'], V_Str = d2_area['right'], Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
        - F0(A_str = d2_area['left'], V_Str = d2_area['left'], Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
    )
    
    #f6 ~ f9 : inflow interploation 
    f6 = ((par_area['left'] * par_vel['left']) + (par_area['right'] * par_vel['right'])) - 2 * (par_area['mid'] * par_vel['mid'])
    f6 = ((d1_area['left'] * d1_vel['left']) + (d1_area['right'] * d1_vel['right'])) - 2 * (d1_area['mid'] * d1_vel['mid'])
    f6 = ((d2_area['left'] * d2_vel['left']) + (d2_area['right'] * d2_vel['right'])) - 2 * (d2_area['mid'] * d2_vel['mid'])

    #f9 ~ f11 : cross section interploation 
    f9 = ((par_area['left']) + (par_area['right'])) - 2(par_area['mid'])
    f10 = ((d1_area['left']) + (d1_area['right'])) - 2(d1_area['mid'])
    f11 = ((d2_area['left']) + (d2_area['right'])) - 2(d2_area['mid'])
    
    #f12 : inflow conservation j = n + 1/2
    #f13 : inflow conservation j = n + 1
    f12 = (d1_area['mid'] * d1_vel['mid']) + (d2_area['mid'] * d2_vel['mid']) - (par_area['mid'] * par_vel['mid'])
    f13 = (d1_area['up'] * d1_vel['up']) + (d2_area['up'] * d2_vel['up']) - (par_area['up'] * par_vel['up'])

    #f14 : equality of pressures p - d1, d2 / j = n + 1/2
    f14 = \
    AtoP(A_str = par_area['mid'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
    - \
    AtoP(A_str = d1_area['mid'], Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)

    f15 = \
    AtoP(A_str = par_area['mid'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu) 
    - \
    AtoP(A_str = d2_area['mid'], Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
    
    #f15 : equality of pressures p - d1, d2 / j = n + 1
    f16 = \
    AtoP(A_str = par_area['up'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
    - \
    AtoP(A_str = d1_area['up'], Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)

    f17 = \
    AtoP(A_str = par_area['up'], Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu) 
    - \
    AtoP(A_str = d2_area['up'], Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
    
    return [ f0, f1, f2, f3, f4, f5, \
    f6, f7, f8, f9, f10, f11, \
    f12, f13, f14, f15, f16, f17 ]  

def get_fun_assigned(fun, variables):
    variables = [ 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', \
    'x6', 'x7', 'x8', 'x9', 'x10', 'x11', \
    'x12', 'x13', 'x14', 'x15', 'x16', 'x17' ]
    return fun(variables[0], variables[1], variables[2], variables[3], variables[4], variables[5],
    variables[6], variables[7], variables[8], variables[9], variables[10], variables[11],
    variables[12], variables[13], variables[14], variables[15], variables[16], variables[17])

def get_residuals_assigned(residuals, variables):
    residuals_assigned = [\
        get_fun_assigned(residual, variables) for residual in residuals \
    ]
    return np.array(residuals_assigned)

def get_Jacobian_assigned(Jacobian, variables):
    Jacobian_assigned = [ [ 0.0 for _ in range(18) ] for _ in range(18) ]
    for y, row in enumerate(Jacobian):
        for x, fun in enumerate(row):
            Jacobian_assigned[y, x] = get_fun_assigned(fun, variables)
    return np.array(Jacobian_assigned)

def get_Jacobian(residuals, variables_dict):
    Jacobian = np.zeros((18, 18)) 
    for y, f in enumerate(residuals):
        for x, v in enumerate(variables):
            Jacobian[y][x] = f.diff(v).subs(variables_dict)
    return Jacobian

def get_assigned(dt, dx,
    U_mesh_p, U_mesh_d1, U_mesh_d2,
    U_mesh_half_p, U_mesh_half_d1, U_mesh_half_d2,
    Rd_p, Rd_d1, Rd_d2,
    E_p, E_d1, E_d2,
    h_p, h_d1, h_d2,
    Pd, rho, mu, solution_variables_vector
):
    residual_functions = get_residual_functions
    variables_syms = [ 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', \
    'x6', 'x7', 'x8', 'x9', 'x10', 'x11', \
    'x12', 'x13', 'x14', 'x15', 'x16', 'x17' ]
    variables_list = [ val for val in solution_variables_vector ]
    variables_dict = zip(variables_sym, variables_list)

    assigned_residual = np.array([ fun.subs(variables_dict) for fun in residual_functions ])
    assigned_Jacobian = np.zeros((18, 18)) 
    for y, f in enumerate(residuals):
        for x, v in enumerate(variables):
            assigned_Jacobian[y][x] = f.diff(v).subs(variables_dict)
    return assigned_residual, assigned_Jacobian

def update_solution_bifu_cond(
    dt, dx,
    U_mesh_p, U_mesh_d1, U_mesh_d2,
    U_mesh_half_p, U_mesh_half_d1, U_mesh_half_d2,
    Rd_p, Rd_d1, Rd_d2,
    E_p, E_d1, E_d2,
    h_p, h_d1, h_d2,
    Pd, rho, mu,
    U_mesh_new_p, U_mesh_new_d1, U_mesh_new_d2
    ):
    k = 0
    solution_variables_vector = np.copy(variables)

    while k < 1000 and np.linalg.norm(solution_variables - update_variables) < 1e-12:
        #get assigned residual
        residuals_assigned, Jacobian_assigned = get_assigned(dt, dx,
        U_mesh_p, U_mesh_d1, U_mesh_d2,
        U_mesh_half_p, U_mesh_half_d1, U_mesh_half_d2,
        Rd_p, Rd_d1, Rd_d2,
        E_p, E_d1, E_d2,
        h_p, h_d1, h_d2,
        Pd, rho, mu, solution_variables_vector)

        inv_Jacobian_assigned = np.linalg.inv(Jacobian_assigned)
        updated_variables = solution_variables - np.matmul(inv_Jacobian_assigned, residuals_assigned)
        k = k + 1

    U_mesh_new_p[0][-1] = solution_variables[9]
    U_mesh_new_d1[0][0] = solution_variables[12]
    U_mesh_new_d2[0][0] = solution_variables[15]

    U_mesh_new_p[1][-1] = solution_variables[0]
    U_mesh_new_d1[1][0] = solution_variables[3]
    U_mesh_new_d2[1][0] = solution_variables[6]

def get_S_F_half_mesh(U_mesh, Q_in, dt, dx, t, T, Rd, Pd, E, rho, h, mu, \
    R1, R2, C
    ):
    nx = U_mesh.shape[1]
    F_mesh = UtoF_mesh(U_mesh = U_mesh, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)
    S_mesh = UtoS_mesh(U_mesh = U_mesh, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)
    U_mesh_half = np.zeros((2,nx-1))
    for i in range(nx-1):
        #half
        U_left_half = U_mesh[:,i]
        U_right_half = U_mesh[:,i+1]

        F_left_half = F_mesh[:,i]
        F_right_half = F_mesh[:,i+1]
        
        S_left_half = S_mesh[:,i]
        S_right_half = S_mesh[:,i+1]
       
        #
        U_half = (U_left_half + U_right_half) / 2
        U_half = U_half + (dt/2) * (F_left_half - F_right_half) / dx
        U_half = U_half + (dt/2) * (S_left_half + S_right_half) / 2

        #
        U_mesh_half[:,i] = U_half
        
        assert U_half[0] >= 0
        assert np.isnan(U_half[0]) == False 
        assert np.isnan(U_half[1]) == False

    F_mesh_half = UtoF_mesh(U_mesh = U_mesh_half, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)
    S_mesh_half = UtoS_mesh(U_mesh = U_mesh_half, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)

    return F_mesh, S_mesh, U_mesh_half, F_mesh_half, S_mesh_half

def get_new_inter_U(U_mesh, Q_in, dt, dx, t, T, Rd, Pd, E, rho, h, mu, \
    R1, R2, C
    ):
    nx = U_mesh.shape[1]
    
    #1. preprocessing : F_mesh, S_mesh, and U_mesh_half
    F_mesh, S_mesh, U_mesh_half, F_mesh_half, S_mesh_half \
    = \
    get_S_F_half_mesh(
        U_mesh = U_mesh, Q_in = Q_in, dt = dt, dx = dx, t = t, T = T, Rd = Rd, 
        Pd = Pd, E = E, rho = rho, h = h, mu = mu, 
        R1 = R1, R2  = R2, C = C
    )

    #mesh new
    U_mesh_new = np.zeros((2,nx))
    
    #intermediate
    for i in range(1, nx-1):
        U_new = U_mesh[:,i] - (dt / dx) * (F_mesh_half[:,i] - F_mesh_half[:,i-1])
        U_new = U_new + (dt / 2) * (S_mesh_half[:,i] + S_mesh_half[:,i-1])
        U_mesh_new[:,i] = U_new
        
        assert U_new[0] >= 0
        assert np.isnan(U_new[0]) == False 
        assert np.isnan(U_new[1]) == False

    return U_mesh_new

def get_new_inlet_U(U_mesh, Q_in, dt, dx, t, T, Rd, Pd, E, rho, h, mu, \
    R1, R2, C
    ):
    nx = U_mesh.shape[1]
    
    #1. preprocessing : F_mesh, S_mesh, and U_mesh_half
    F_mesh, S_mesh, U_mesh_half, F_mesh_half, S_mesh_half \
    = \
    get_S_F_half_mesh(
        U_mesh = U_mesh, Q_in = Q_in, dt = dt, dx = dx, t = t, T = T, Rd = Rd, 
        Pd = Pd, E = E, rho = rho, h = h,  mu = mu, 
        R1 = R1, R2 = R2, C = C
    )

    #inlet
    #Q_middle = Q_in(t - dt/2 if t - dt/2 < T else t - dt/2 - T)  
    Q_middle = Q_in((t - dt/2) % T)
    Q_right = U_mesh_half[0][0] * U_mesh_half[1][0]
    Q_left = 2 * Q_middle - Q_right

    A_previous = U_mesh[0][0]
    assert A_previous >= 0
    A_new = A_previous - (dt / dx) * (Q_right - Q_left)
    assert A_new >= 0

    #Q_new = Q_in(t if t < T else t - T)
    Q_new = Q_in(t % T)
    V_new = QtoV(Q = Q_new, A = A_new, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)
    U_in = np.array([A_new, V_new])

    assert np.isnan(U_in[0]) == False 
    assert np.isnan(U_in[1]) == False

    return U_in

def get_new_outlet_U(U_mesh, U_mesh_new, Q_in, dt, dx, t, T, Rd, Pd, E, rho, h, mu,
    R1, R2, C
    ):
    nx = U_mesh.shape[1]
    #outlet
    A_new_left = U_mesh_new[0][nx-2] 
    V_new_left = U_mesh_new[1][nx-2]
    Q_new_left = VtoQ(V = V_new_left, A = A_new_left, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)
    
    A_previous_middle = U_mesh[0][nx-1]
    
    P_previous_middle = AtoP(A = A_previous_middle, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)
    
    V_previous_middle = U_mesh[1][nx-1]
    
    Q_previous_middle = VtoQ(V = V_previous_middle, A = A_previous_middle, \
    Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu)

    P_new_middle = P_previous_middle

    for k in range(1000):
        P_old = P_previous_middle

        #Q_new_middle
        Q_new_middle = Q_previous_middle + (P_new_middle - P_previous_middle) / R1
        Q_new_middle = Q_new_middle + (dt/(R1*R2*C)) * (P_previous_middle - Q_previous_middle * (R1 + R2))
        
        #A_new_middle
        A_new_middle = A_previous_middle - (dt / dx) * (Q_new_middle - Q_new_left)
        P_new_middle = AtoP(A = A_new_middle,
        Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu)

        if abs(P_old - P_new_middle) < 1e-3:
            break

    V_new_middle = QtoV(Q = Q_new_middle, A = A_new_middle, \
    Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu)

    assert A_new_middle is not np.nan
    assert V_new_middle is not np.nan
        
    return np.array([A_new_middle, V_new_middle])

def show(time_mid, pressure_mid, inflow_mid, tf, dt):
    plt.subplot(211)
    plt.title('pressure')
    plt.plot(time_mid, pressure_mid)
    plt.grid()

    plt.subplot(212)
    plt.title('inflow')
    plt.plot(time_mid, inflow_mid)

    plt.savefig('multi2.png', dpi = 300)

def bifurcation_simulation(
    Rds, Es, Ls, hs, rho, Pd, mu, dx, #shape ) unit : m
    T, dt, tc, tf, #time ) unit : second
    R1, R2, C
):
    # 0 / 1 2 / 3 4 5 6 / 7 8 9 10 11 12 13 14
    # 1 / 2 3 / 4 5 6 7 / 8 9 10 11 12 13 14 15
    U_meshes = []

    for idx, shape in enumerate(zip(Rds, Es, Ls, hs)):
        Rd, E, L, h = shape
        depth = int(np.log2(idx + 1))
        width = 2**depth
        nx = int(L/dx) + 1
        Q_in = inlet(Rd)
        
        print('depth', depth)

        U_mesh = np.zeros((2,nx))
        for i in range(nx):
            U_mesh[0][i] = np.pi * Rd * Rd
            U_mesh[1][i] = Q_in(0) / width
        U_meshes.append(U_mesh)

    #process
    time_mid = []
    pressure_mid = []
    inflow_mid = []
    t = 0

    #print_status = True
    print_status = False

    iterations = int(tf / dt)

    for iteration in tqdm(range(iterations)):  
        if iteration % 100 == 0 and iteration != 0:
            print('iteration / iterations and ratio', iteration, iterations, iteration/iterations)
            show(time_mid = time_mid, pressure_mid = pressure_mid, inflow_mid = inflow_mid, tf = tf, dt = dt)
        t = t + dt
        U_meshes_new = []

        if print_status == True:
            print('iteration', iteration)

        #1. intermediate
        if print_status == True:
            print("1")

        for idx, shape in enumerate(zip(Rds, Es, Ls, hs)):
            Rd, E, L, h = shape
            U_mesh = U_meshes[idx]
            nx = U_mesh.shape[1]
            mid = int(nx/2)
            
            #make U_new
            U_mesh_new = get_new_inter_U(U_mesh = U_mesh, Q_in = Q_in,
            dt = dt, dx = dx, t = t, T = T,
            Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu,
            R1 = R1, R2 = R2, C = C)
            U_meshes_new.append(U_mesh_new)

        #2. inlet ( only for idx == 0 )   
        if print_status == True:
            print("2")
        U_meshes_new[0][:,0] = get_new_inlet_U(U_mesh = U_meshes[0], Q_in = Q_in, \
        dt = dt, dx = dx, t = t, T = T, \
        Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu, \
        R1 = R1, R2 = R2, C = C)

        #3. outlet
        if print_status == True:
            print("3")
        for idx, shape in enumerate(zip(Rds, Es, Ls, hs)):
            if 2 * idx + 1 >= len(Rds):
                Rd, E, L, h = shape
                U_mesh = U_meshes[idx]
                nx = U_mesh.shape[1]
                mid = int(nx/2)

                U_meshes_new[idx][:,-1] = get_new_outlet_U(U_mesh = U_meshes[idx], U_mesh_new = U_meshes_new[idx], Q_in = Q_in, \
                dt = dt, dx = dx, t = t, T = T, \
                Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu, \
                R1 = R1, R2 = R2, C = C)

        #5. store the intermediate value
        if print_status == True:
            print("5")
        for idx, shape in enumerate(zip(Rds, Es, Ls, hs)):
            Rd, E, L, h = shape
            U_mesh = U_meshes[idx]
            nx = U_mesh.shape[1]
            mid = int(nx/2)
            time_mid.append(t) 
            A = U_mesh[0][mid]
            V = U_mesh[1][mid]
            pressure_mid.append(AtoP(A = A, Rd = Rd, Pd = Pd, E = E, \
            rho = rho, h = h, mu = mu))
            inflow_mid.append(VtoQ(V = V, A = A, Rd = Rd, Pd = Pd, E = E, \
            rho = rho, h = h, mu = mu))

        #6. check wheter value are nan or not
        if print_status == True:
            print("6")
        for idx, shape in enumerate(zip(Rds, Es, Ls, hs)):
            Rd, E, L, h = shape
            U_mesh = U_meshes_new[idx]
            nx = U_mesh.shape[1]
            for i in range(nx):
                assert np.isnan(U_mesh[0][i]) == False
                assert np.isnan(U_mesh[1][i]) == False
                assert U_mesh[0][i] > 0
                assert U_mesh[1][i] > 0

        #7. update meshes
        if print_status == True:
            print("7")
            print('U meshes', len(U_meshes))
            print(U_meshes)
            print('U meshes new', len(U_meshes_new))
            print(U_meshes_new)

        for idx, U_mesh_new in enumerate(U_meshes_new):
            U_meshes[idx] = np.copy(U_mesh_new)

        continue

        #4. bifurcation
        if print_status == True:
            print("4")

    #print result
    """
    time_mid = np.array(time_mid)
    pressure_mid = np.array(pressure_mid)
    inflow_mid = np.array(inflow_mid)
    """

    show(time_mid = time_mid, pressure_mid = pressure_mid, inflow_mid = inflow_mid, tf = tf, dt = dt)

if __name__=="__main__":
    #shape ) unit : m
    Rds = [ 0.3 * 1e-2 ] #0.3cm
    Es = [ 700 * 1e3 ] 
    Ls = [ 12.6 * 1e-2 ]
    hs = [ 0.3 * 1e-3 ]
    rho = 1060 
    Pd = 10.9 * 1e3
    mu = 4 * 1e-3
    dx = 0.1 * 1e-2

    #time ) unit : second
    T = 1.1 #1.1s
    dt = 1e-5 #1e-5s
    tc = 1.2
    tf = tc * T

    #for 3WK modelq
    R1 = 2.4875 * 1e8 #Pa s m-3
    R2 = 1.8697 * 1e9 #Pa s m-3 
    C = 1.7529 * 1e-10 #m^3 Pa-1

    bifurcation_simulation(
        Rds = Rds, Es = Es, Ls = Ls, hs = hs, 
        rho = rho, Pd = Pd, mu = mu, dx = dx, #shape ) unit : m
        T = T, dt = dt, tc = tc, tf = tf, #time ) unit : second
        R1 = R1, R2 = R2, C = C
    )