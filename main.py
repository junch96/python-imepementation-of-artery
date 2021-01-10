import artery

def common_carotid():
    #shape ) unit : m
    Rds = [ 0.3 * 1e-2 ] #0.3cm
    Es = [ 700 * 1e3 ] 
    Ls = [ 12.6 * 1e-2 ]
    hs = [ 0.3 * 1e-3 ]
    rho = 1060 
    Pd = 10.9 * 1e3
    mu = 4 * 1e-3
    dx = 0.5 * 1e-2

    #time ) unit : second
    T = 0.01 #1.1s
    dt = 1e-5 #1e-5s
    tc = 1.2
    tf = tc * T

    #for 3WK modelq
    R1 = 2.4875 * 1e8 #Pa s m-3
    R2 = 1.8697 * 1e9 #Pa s m-3 
    C = 1.7529 * 1e-10 #m^3 Pa-1

    inlet_indices = [ 0 ]
    joint_indices = [ ]
    outlet_indices = [ 0 ]

    artery.bifurcation_simulation(
        Rds = Rds, Es = Es, Ls = Ls, hs = hs, 
        rho = rho, Pd = Pd, mu = mu, dx = dx, #shape ) unit : m
        T = T, dt = dt, tc = tc, tf = tf, #time ) unit : second
        R1 = R1, R2 = R2, C = C,
        inlet_indices = inlet_indices,
        outlet_indices = outlet_indices,
        joint_indices = joint_indices,
        file_inlet = "data/common_carotid.csv"
    )

def aortic_bifurcation():
    #section 3.7
    #table 3
    #Figure 11

    #shape ) unit : m
    Rds = [ 0.86 * 1e-2, 0.6 * 1e-2, 0.6 * 1e-2 ]
    Es = [ 500 * 1e3, 700 * 1e3, 700 * 1e3 ] 
    Ls = [ 8.6 * 1e-2, 8.5 * 1e-2, 8.5 * 1e-2 ]
    hs = [ 1.032 * 1e-3, 0.72 * 1e-3, 0.72 * 1e-3 ]
    rho = 1060 
    Pd = 9.5 * 1e3
    mu = 4 * 1e-3
    dx = 0.1 * 1e-2

    #time ) unit : second
    T = 1.1 #1.1s
    dt = 1e-4 #1e-5s
    tc = 2.1
    tf = tc * T

    #for 3WK modelq
    R1 = 6.8123 * 1e7 #Pa s m-3
    R2 = 3.1013 * 1e9 #Pa s m-3 
    C = 3.6664 * 1e-10 #m^3 Pa-1

    inlet_indices = [ 0 ]
    joint_indices = [ (0, 1, 2) ]
    outlet_indices = [ 1, 2 ]

    artery.bifurcation_simulation(
        Rds = Rds, Es = Es, Ls = Ls, hs = hs, 
        rho = rho, Pd = Pd, mu = mu, dx = dx, #shape ) unit : m
        T = T, dt = dt, tc = tc, tf = tf, #time ) unit : second
        R1 = R1, R2 = R2, C = C,
        inlet_indices = inlet_indices,
        outlet_indices = outlet_indices,
        joint_indices = joint_indices,
        file_inlet = "data/aortic_bifurcation.csv"
    )

if __name__=="__main__":
    common_carotid()
    #aortic_bifurcation()
