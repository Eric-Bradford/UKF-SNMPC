# SNMPC model and problem setup
import numpy as np
from casadi import *

def specifications():
    
    # NMPC algorithm
    tf = 2.                  # time horizon
    nk = 20                  # number of control intervals 
    shrinking_horizon = True # shrinking horizon or receding horizon
    robust_horizon    = 1    # Number of control intervals for which covariance is propagated
        
    # Discredization using direct collocation 
    deg  = 5       # Degree of interpolating polynomialc
    cp   = "radau" # Type of collocation points
    nicp = 1       # Number of (intermediate) collocation points per control interval
    
    # Simulation
    simulation_time   = 2. # simulation time
    number_of_repeats = 10 # number of Monte Carlo simulations for verification  
    
    # Unscented Kalman filter specifications
    alpha = 0.4  # alpha
    beta  = 2.   # beta
    kappa = 1.   # kappa
    
    # NLP solver
    opts                                = {}
    opts["expand"]                      = True
    opts["ipopt.max_iter"]              = 10000
    opts["ipopt.tol"]                   = 1e-8
    opts['ipopt.linear_solver']         = 'ma27'
    opts["ipopt.warm_start_init_point"] = "yes"
          
    return tf, nk, shrinking_horizon, deg, cp, nicp, simulation_time, opts, \
 number_of_repeats, alpha, beta, kappa, robust_horizon

def DAE_system():
    
    tf, nk, shrinking_horizon, deg, cp, nicp, simulation_time, opts, \
    number_of_repeats, alpha, beta, kappa, robust_horizon = specifications()
    
    # Define vectors with names of states
    states     = ['CA','CB','CC','T','Vol'] 
    nd         = len(states)
    xd         = MX.sym('xd',nd)  
    for i in range(nd):
        globals()[states[i]] = xd[i]
    
    # Define vectors with names of algebraic variables
    algebraics = ['r1','r2'] 
    na         = len(algebraics)
    xa         = MX.sym('xa',na)  
    for i in range(na):
        globals()[algebraics[i]] = xa[i]
    
    # Define vectors with banes of input variables
    inputs     = ['F','T_a']
    nu         = len(inputs)
    u          = MX.sym("u",nu)
    for i in range(nu):
        globals()[inputs[i]] = u[i]
    
    # Define model parameter names and values
    modpar    = ['CpA','CpB','CpC','CpH2SO4','T0','HRA','HRB','E1A','E2A','A1',\
                 'Tr1','Tr2']
    modparval = [30.,60.,20.,35.,305.,-6500.,8000.,9500./1.987,7000./1.987,\
                 1.25,420.,400.]
    nmp       = len(modpar)
    for i in range(nmp):
        globals()[modpar[i]] = MX(modparval[i])
    
    # Uncertain parameter names with mean and covariance matrix 
    unpar = ['CA0','A2','UA','N0H2S04']
    nun   = len(unpar)
    MeanP = np.array([4.,0.08,4.5,100.])
    CovP  = [0.1,0.00016,0.2,5.]*diag(np.ones(nun))
    xu    = MX.sym('xu',nun)  
    for i in range(nun):
        globals()[unpar[i]] = MX(xu[i])
    
    # Actual mean and covariance of differential states at t = 0        
    Meanx0   = np.array([0.5,1.,0.1,290.,100.])
    Covx0    = [1e-4,1e-4,2e-4,0.1,0.2]*diag(np.ones(nd))
    
    # Previous control input and previous state estimate at t = 0
    u0       = np.array([250.,500.])
    MeanSEx0 = np.array([0.5,1.,0.1,290.,100.])
    CovSEx0  = [1e-4,1e-4,2e-4,0.1,0.2]*diag(np.ones(nd))
    
    # "Common" values for states and algebraics to help solver converge
    xD_init = np.array([1.,0.2,1.,300.,400.])
    xA_init = np.array([0.01,1e-4])
    u_init  = np.array([100.,400.])
    
    # Additive disturbance noise 
    Sigma_w = [1e-4,1e-4,2e-4,0.1,0.2]*diag(np.ones(nd))
    
    # Declare ODE equations (use notation as defined above) 
    dCA   = -r1*CA + (CA0-CA)*(F/Vol)
    dCB   =  r1*CA/2 - r2*CB - CB*(F/Vol)
    dCC   =  3*r2*CB - CC*(F/Vol)
    dT    =  (UA*10.**4*(T_a-T) - CA0*F*CpA*(T-T0) + (HRA*(-r1*CA)+HRB*(-r2*CB\
    ))*Vol)/((CA*CpA+CpB*CB+CpC*CC)*Vol + N0H2S04*CpH2SO4)
    dVol  =  F
    ODEeq =  [dCA,dCB,dCC,dT,dVol]

    # Declare algebraic equations 
    Aeq = [r1 - A1*exp(E1A*(1./Tr1-1./T)), r2 - A2*exp(E2A*(1./Tr2-1./T))]

    # Define control bounds
    u_min      = np.array([0.  ,270.]) # lower bound of inputs
    u_max      = np.array([250.,500.]) # upper bound of inputs
 
    # Define objective (in expectation) to be minimized
    t           = MX.sym('t')
    Obj_M       = Function('mayer',[xd,u],[-CC*Vol]) # Mayer term
    Obj_L       = Function('lagrange',[xd,u],[0.])   # Lagrange term
    R           = [2e-4,5e-5]*diag(np.ones(nu))          # Control change penality 
     
    # Define path constraint functions g(x) <= 0
    gpdef      = vertcat(T-420.,Vol-800.)       # g(x)
    ngp        = MX.size(gpdef)[0]              # Number of constraints
    gpfcn      = Function('gpfcn',[xd],[gpdef]) # Function definition
    pgp        = MX([0.05,0.05])                # Probability of constraint violation
    
    # Define terminal constraint functions g(x) <= 0
    gtdef = vertcat(MX())                  # g(x) 
    ngt   = MX.size(gtdef)[0]              # Number of constraints                  
    gtfcn = Function('gtfcn',[xd],[gtdef]) # Function definition
    pgt   = MX([])                         # Probability of constraint violation
    
    # Measurement model
    measfcn = vertcat(CA,CB,T)                   # h(x) 
    nm      = MX.size(measfcn)[0]                # Number of measurements  
    hfcn    = Function('hfcn',[xd,xu],[measfcn]) # Function definition
    Sigma_v = [1e-3,1e-3,1e-2]*diag(np.ones(nm))   # Measurements noise covariance matrix
    
    return xd, xa, xu, u, ODEeq, Aeq, Obj_M, Obj_L, R, u_min, \
u_max, states, algebraics, inputs, hfcn, Sigma_v, ngp, gpfcn, pgp, \
 Sigma_w, ngt, gtfcn, pgt, CovP, MeanP, nm, Meanx0, Covx0, MeanSEx0, CovSEx0, u0,\
 xD_init, xA_init, u_init     