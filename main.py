#sci package
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#sympy
from sympy import *

#vampy
import vampy

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
    #thickness
    #h = 0.3 * 1e-3 #0.3mm

    #Ad
    Ad = Rd * Rd * np.pi

    #E : Young's modulus
    #E = 700 * 1e3 # 700kPa
    
    #beta
    beta = (4/3)*(np.sqrt(np.pi)*E*h)

    #Pd
    #Pd = 10.9 * 1e3

    re = Pd + (beta / Ad) * (np.sqrt(A) - np.sqrt(Ad))
    #print('re',re)
    assert Ad >= 0
    assert np.isnan(re) == False
    return re

#dU/dt + dF/dZ = S

def UtoF(U, Rd, Pd, E, rho, h, mu):
    #rho
    #rho = 1060

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
    #rho = 1060
    
    #mu
    #mu = 4 * 1e-3

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
    #print('Q',Q)
    #print('A',A)
    re = Q / A
    #print('re',re)
    assert np.isnan(re) == False
    return re

def VtoQ(V, A, Rd, Pd, E, rho, h, mu):
    #print('V',V)
    #print('A',A)
    re = V * A
    #print('re',re)
    assert np.isnan(re) == False
    return re


def get_residual_functions(dt, dx, \
    U_mesh_p, U_mesh_d1, U_mesh_d2, \
    U_mesh_half_p, U_mesh_half_d1, U_mesh_half_d2, \
    Rd_p, Rd_d1, Rd_d2, \
    Pd, 
    E_p, E_d1, E_d2,  
    rho, 
    h_p, h_d1, h_d2, 
    mu):
    Ad_p = np.pi * Rd_p * Rd_p
    Ad_d1 = np.pi * Rd_d1 * Rd_d1
    Ad_d2 = np.pi * Rd_d2 * Rd_d2

    #U F S 

    #U[0] : area
    #U[1] : average velocity     

    x0 = Symbol('x0')
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    x4 = Symbol('x4')
    x5 = Symbol('x5')
    
    x6 = Symbol('x6')
    x7 = Symbol('x7')
    x8 = Symbol('x8')
    x9 = Symbol('x9')
    x10 = Symbol('x10')
    x11 = Symbol('x11')
    
    x12 = Symbol('x12')
    x13 = Symbol('x13')
    x14 = Symbol('x14')
    x15 = Symbol('x15')
    x16 = Symbol('x16')
    x17 = Symbol('x17')

    def AtoP(A_str, Rd, Pd, E, rho, h, mu):
        A = Symbol(A_str)
        beta = (4/3)*(np.sqrt(np.pi)*E*h)
        re = Pd + (beta / Ad) * (np.sqrt(A) - np.sqrt(Ad))
        return P

    def F0(A_str, V_str, Rd, Pd, E, rho, h, mu):
        A = Symbol(A_str)
        V = Symbol(V_str)
        re = A * V
        return re

    def F1(A_str, V_str, Rd, Pd, E, rho, h, mu):
        A = Symbol(A_str)
        V = Symbol(V_str)
        P = AtoP(A_str, Rd, Pd, E, rho, h, mu)
        re = 0.5 * V * V + (1/rho) * P
        return re
    
    def S1(A_str, V_str, Rd, Pd, E, rho, h, mu):
        A = symbol(A_str)
        V = symbol(V_str)
        f = -8 * mu * np.pi * V
        re = f / (rho * A)
        return re

    U_prev_p = U_mesh_p[:,-1]
    U_mid_p = U_mesh_half_p[:,-1]

    U_prev_d1 = U_mesh_d1[:,0]
    U_mid_d1 = U_mesh_half_d1[:,0]
    
    U_prev_d2 = U_mesh_d2[:,0]
    U_mid_d2 = U_mesh_half_d2[:,0]

    #f0 ~ f2 : navier second term
    f0 = (x0 - U_prev_p[1]) 
    + (dt / dx)
    (
        F1(A_str = 'x11', V_Str = 'x2', Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
         - UtoF(U = U_mid_p, Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
    ) 
    - (dt / 2)
    (
        S1(A_str = 'x11', V_str = 'x2', Rd = Rd_p, Pd = Pd, E = E_p, rho = rho,h = h_p, mu = mu) 
         + UtoS(U = U_mid_p, Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
    )
    f1 = (x3 - U_prev_d1[1]) 
    + (dt / dx)
    (
        UtoF(U = U_mid_d1, Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
        - F1(A_str = 'x14', V_Str = 'x5', Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    ) 
    - (dt / 2)
    (
        UtoS(U = U_mid_d1, Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
        - S1(A_str = 'x14', V_str = 'x5', Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    )
    f2 = (x6 - U_prev_d2[1]) 
    + (dt / dx)
    (
        UtoF(U = U_mid_d2, Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
        - F1(A_str = 'x17', V_Str = 'x8', Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
    ) 
    - (dt / 2)
    (
        UtoS(U = U_mid_d2, Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
        + S1(A_str = 'x17', V_str = 'x8', Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu) 
    )
    #f3 ~ f5 : navier first term
    f3 = (x9 - U_prev_p[0]) 
    + (dt / dx)
    (
        F0(A_str = 'x11', V_Str = 'x2', Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
        - UtoF(U = U_mid_p, Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
    )
    f4 = (x12 - U_prev_d1[0]) 
    + (dt / dx)
    ( 
        UtoF(U = U_mid_d1, Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
        - F0(A_str = 'x14', V_Str = 'x5', Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    )
    f5 = (x15 - U_prev_d2[0]) 
    + (dt / dx)
    ( 
        UtoF(U = U_mid_d2, Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
        - F0(A_str = 'x17', V_Str = 'x8', Rd = Rd_d2, Pd = Pd, E = E_d2, rho = rho, h = h_d2, mu = mu)
    )
    
    #f6 ~ f9 : inflow interploation 
    f6 = 2 * (x11 * x2) - (x10 * x1)  + (U_mid_p[0] * U_mid_p[1]) 
    f7 = 2 * (x14 * x5) - (x13 * x4)  + (U_mid_d1[0] * U_mid_d1[1])
    f8 = 2 * (x17 * x8) - (x16 * x7)  + (U_mid_d2[0] * U_mid_d2[1])

    #f9 ~ f11 : cross section interploation 
    f9 = 2 * (x11) - (x10)  + U_mid_p[0]
    f10 = 2 * (x14) - (x13)  + U_mid_d1[0] 
    f11 = 2 * (x17) - (x16)  + U_mid_d2[0]
    
    #f12 : inflow conservation j = n + 1/2
    #f13 : inflow conservation j = n + 1
    f12 = (x10 * x1) - (x13 * x4) - (x16 * x7)
    f13 = (x9 * x0) - (x12 * x3) - (x15 * x6)

    #f14 : equality of pressures p - d1, d2 / j = n + 1/2
    f14 = \
    AtoP(A_str = 'x10', Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu)
    - \
    AtoP(A_str = 'x13', Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    f15 = \
    AtoP(A_str = 'x10', Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu) 
    - \
    AtoP(A_str = 'x16', Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    
    #f15 : equality of pressures p - d1, d2 / j = n + 1
    f14 = \
    AtoP(A_str = 'x9', Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu) 
    - \
    AtoP(A_str = 'x12', Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    f15 = \
    AtoP(A_str = 'x9', Rd = Rd_p, Pd = Pd, E = E_p, rho = rho, h = h_p, mu = mu) 
    - \
    AtoP(A_str = 'x15', Rd = Rd_d1, Pd = Pd, E = E_d1, rho = rho, h = h_d1, mu = mu)
    
    return [ f0, f1, f2, f3, f4, f5, \
    f6, f7, f8, f9, f10, f11, \
    f12, f13, f14, f15, f16, f17 ]  

def get_Jacobian(residuals):
    variables = [ 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', \
    'x6', 'x7', 'x8', 'x9', 'x10', 'x11', \
    'x12', 'x13', 'x14', 'x15', 'x16', 'x17' ]
    Jacobian = np.zeros((18, 18)) 
    for y, f in enumerate(residuals):
        for x, v in enumerate(variables):
            Jacobian[y][x] = f.diff(v)
    return Jacobian

def get_fun_assigned(fun, variables):
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

def get_solution_bif_cond(variables, residuals, Jacobian):
    k = 0
    
    solution_variables = np.copy(variables)

    while k < 1000 and np.linalg.norm(solution_variables - update_variables) < 1e-12:
        residuals_assigned = []
        Jacobian_assigned = []
        inv_Jacobian_assigned = np.linalg.inv(Jacobian_assigned)

        update_variables = solution_variables - np.matmul(inv_Jacobian_assigned, residuals_assigned)

    return solution_variables

def get_new_U(U_mesh, Q_in, dt, dx, t, T, Rd, Pd, E, rho, h, mu, \
    R1, R2, C
    ):
    #
    nx = U_mesh.shape[1]
    
    #S and F
    F_mesh = UtoF_mesh(U_mesh = U_mesh, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)
    
    S_mesh = UtoS_mesh(U_mesh = U_mesh, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)
    
    #print(U_mesh.shape)
    #print(F_mesh.shape)
    #print(S_mesh.shape)
    
    #U_half
    U_mesh_half = np.zeros((2,nx-1))
    
    for i in range(nx-1):
        #half
        U_left_half = U_mesh[:,i]
        U_right_half = U_mesh[:,i+1]
        #print(U_left_half.shape)
        #print(U_right_half.shape)

        F_left_half = F_mesh[:,i]
        F_right_half = F_mesh[:,i+1]
        #print(F_left_half.shape)
        #print(F_right_half.shape)
        
        S_left_half = S_mesh[:,i]
        S_right_half = S_mesh[:,i+1]
        #print(S_left_half.shape)
        #print(S_right_half.shape)
       
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
        
    #inlet
    #Q_middle = Q_in(t - dt/2 if t - dt/2 < T else t - dt/2 - T)  
    Q_middle = Q_in((t - dt/2) % T)
    Q_right = U_mesh_half[0][0] * U_mesh_half[1][0]
    Q_left = 2 * Q_middle - Q_right

    A_previous = U_mesh[0][0]
    assert A_previous >= 0
    A_new = A_previous - (dt / dx) * (Q_right - Q_left)
    #print(A_new)
    assert A_new >= 0

    #Q_new = Q_in(t if t < T else t - T)
    Q_new = Q_in(t % T)
    V_new = QtoV(Q = Q_new, A = A_new, Rd = Rd, Pd = Pd, E = E, \
    rho = rho, h = h, mu = mu)
    U_in = np.array([A_new, V_new])

    U_mesh_new[:,0] = U_in
    assert np.isnan(U_in[0]) == False 
    assert np.isnan(U_new[1]) == False
       
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
        #print(Rd, dt / dx, Q_new_left, Q_previous_middle)
        A_new_middle = A_previous_middle - (dt / dx) * (Q_new_middle - Q_new_left)
        #print('A_new_middle', A_new_middle)
        P_new_middle = AtoP(A = A_new_middle,
        Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu)

        if abs(P_old - P_new_middle) < 1e-3:
            break

    assert A_new_middle is not np.nan
    V_new_middle = QtoV(Q = Q_new_middle, A = A_new_middle, \
    Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu)
    assert V_new_middle is not np.nan
        
    U_mesh_new[:,nx-1] = np.array([A_new_middle, V_new_middle])

    return U_mesh_new

def show(time_mid, pressure_mid, inflow_mid, tf, dt):
    #time_x = np.arange(0, tf, dt)
    #inflow_interp1d = interp1d(time_mid, inflow_mid)
    #pressure_interp1d = interp1d(time_mid, pressure_mid, kind='linear', bounds_error=False, fill_value = inflow_mid[0])
    #return interp1d(t, v, kind='linear', bounds_error=False, fill_value=v[0])
    
    plt.subplot(211)
    plt.title('pressure')
    plt.plot(time_mid, pressure_mid)
    plt.grid()

    plt.subplot(212)
    plt.title('inflow')
    plt.plot(time_mid, inflow_mid)

    plt.savefig('multi.png', dpi = 300)

def main(
    Rds, Es, Ls, hs, rho, Pd, mu, dx, #shape ) unit : m
    T, dt, tc, tf, #time ) unit : second
    R1, R2, C
):

    U_meshes = []
    # 0 / 1 2 / 3 4 5 6 / 7 8 9 10 11 12 13 14
    # 1 / 2 3 / 4 5 6 7 / 8 9 10 11 12 13 14 15
    # 0 / 

    for idx, shape in enumerate(zip(Rds, Es, Ls, hs)):
        Rd, E, L, h = shape
        depth = int(np.log2(idx + 1))
        width = 2**depth
        #shape ) unit : m
        """
        Rd = 0.3 * 1e-2 #0.3cm
        E = 700 * 1e3 
        L = 12.6 * 1e-2 #12.6cm
        h = 0.3 * 1e-3
        """
    
        """
        rho = 1060 
        Pd = 10.9 * 1e3
        mu = 4 * 1e-3
        dx = 0.1 * 1e-2
        """

        #time ) unit : second
        """
        T = 1.1 #1.1s
        dt = 1e-5 #1e-5s
        tc = 2
        tf = tc * T
        """

        nx = int(L/dx) + 1
        print('nx',nx)
        Q_in = inlet(Rd)
        
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

    iterations = int(tf/dt) 
    print(iterations)
    for iteration in tqdm(range(iterations)):  
        #print('iteration',iteration)
        if iteration % 100 == 0:
            print('iteration / iterations and ratio', iteration, iterations, iteration/iterations)
        t = t + dt
        U_meshes_new = []

        for idx, shape in enumerate(zip(Rds, Es, Ls, hs)):
            Rd, E, L, h = shape
            U_mesh = U_meshes[idx]
            nx = U_mesh.shape[1]
            mid = int(nx/2)
            
            #make U_new
            U_mesh_new = get_new_U(U_mesh = U_mesh, Q_in = Q_in, \
            dt = dt, dx = dx, t = t, T = T, \
            Rd = Rd, Pd = Pd, E = E, rho = rho, h = h, mu = mu, \
            R1 = R1, R2 = R2, C = C)
            U_meshes_new.append(U_mesh_new)

            for i in range(nx):
                assert np.isnan(U_mesh[0][i]) == False
                assert np.isnan(U_mesh[1][i]) == False
                assert U_mesh[0][i] >= 0

            #select
            time_mid.append(t) 
            A = U_mesh[0][mid]
            V = U_mesh[1][mid]
            pressure_mid.append(AtoP(A = A, Rd = Rd, Pd = Pd, E = E, \
            rho = rho, h = h, mu = mu))
            inflow_mid.append(VtoQ(V = V, A = A, Rd = Rd, Pd = Pd, E = E, \
            rho = rho, h = h, mu = mu))

        for idx, U_mesh_new in enumerate(U_meshes_new):
            U_meshes[idx] = U_mesh_new

    #print result
    time_mid = np.array(time_mid)
    pressure_mid = np.array(pressure_mid)
    inflow_mid = np.array(inflow_mid)

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
    tc = 2.0
    tf = tc * T

    #for 3WK modelq
    R1 = 2.4875 * 1e8 #Pa s m-3
    R2 = 1.8697 * 1e9 #Pa s m-3 
    C = 1.7529 * 1e-10 #m^3 Pa-1

    main(
        Rds = Rds, Es = Es, Ls = Ls, hs = hs, 
        rho = rho, Pd = Pd, mu = mu, dx = dx, #shape ) unit : m
        T = T, dt = dt, tc = tc, tf = tf, #time ) unit : second
        R1 = R1, R2 = R2, C = C
    )
"""
def main(
    Rds, Es, Ls, hs, rho, Pd, mu, dx, #shape ) unit : m
    T, dt, tc, tf, #time ) unit : second
    R1, R2, C
):
"""