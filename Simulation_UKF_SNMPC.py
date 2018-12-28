# Closed-loop simulation of defined problem
from UKF_SNMPC import *
import numpy as np
from casadi import *

UKF_SNMPC                = UKF_SNMPC()
solver                   = UKF_SNMPC.solver
U_pasts, Xd_pasts, Xa_pasts, Conp_pasts, Cont_pasts, xu_pasts, MeanSEx_pasts,\
CovSEx_pasts, t_pasts    = UKF_SNMPC.initialization()
number_of_repeats        = UKF_SNMPC.number_of_repeats
simulation_time, nun, nd = UKF_SNMPC.simulation_time, UKF_SNMPC.nun, UKF_SNMPC.nd
Sigma_v, nm, Sigma_w     = UKF_SNMPC.Sigma_v, UKF_SNMPC.nm, UKF_SNMPC.Sigma_w
MeanP, CovP, hfcn        = UKF_SNMPC.MeanP, UKF_SNMPC.CovP, UKF_SNMPC.hfcn
update_inputs, deltat    = UKF_SNMPC.update_inputs, UKF_SNMPC.deltat
cfcn, xhatfcn, Sigmafcn  = UKF_SNMPC.cfcn, UKF_SNMPC.x_hatfcn, UKF_SNMPC.Sigmafcn

for un in range(number_of_repeats):
    ws = np.zeros((UKF_SNMPC.nk,nun+nd))
    arg, u_past, tk, t0i, tfi, u_nmpc, Meanx0, Covx0, MeanSEx, CovSEx, t_past \
               = UKF_SNMPC.initialization_loop()
    xd_current = np.expand_dims(np.random.multivariate_normal(\
              np.array(Meanx0),Covx0),0).T
    xu_current = np.expand_dims(np.random.multivariate_normal(\
              np.array(MeanP),CovP),0).T                                                       
    Xd_pasts[0,un,:]       = np.array(DM(xd_current)).flatten()
    xu_pasts[un,:]         = np.array(DM(xu_current)).flatten()
    MeanSEx_pasts[0,un,:]  = np.array(DM(MeanSEx)).flatten()
    CovSEx_pasts[0,un,:,:] = np.array(DM(CovSEx))
    
    while True:
        
        # Break when simulation time is reached
        if tk >= UKF_SNMPC.nk-1:
            break        
                
        # Simulation and measurement of plant 
        xd_current, xa_current = UKF_SNMPC.simulator(xd_current,u_nmpc,\
                                                    t0i,tfi,xu_current)
        w = np.random.multivariate_normal(np.zeros(nd),Sigma_w)
        xd_current = (xd_current.flatten() + w)
        yd         = DM(hfcn(xd_current,xu_current))  + \
        np.random.multivariate_normal(np.zeros(nm),Sigma_v)
        tfi       += deltat
        
        # Parameter to set initial condition of NMPC algorithm and update discrete time tk
        p, tk    = update_inputs(yd,tk,u_nmpc,MeanSEx,CovSEx)        
        arg["p"] = p
        
        # Solve SNMPC problem and extract results
        res     = solver(**arg)
        u_nmpc  = cfcn(np.array(res["x"])[:,0])     # control input
        MeanSEx = xhatfcn(np.array(res["x"])[:,0])  # mean of state estimate
        CovSEx  = Sigmafcn(np.array(res["x"])[:,0]) # covariance of state estimate
        
        # Collect data
        MeanSEx_pasts[tk+1,un,:]  = np.array(DM(MeanSEx)).flatten()
        CovSEx_pasts[tk+1,un,:,:] = np.array(DM(CovSEx))
        t0i           += deltat
        t_past, u_past = UKF_SNMPC.collect_data(t_past,u_past,t0i,u_nmpc)
        
    # Generate data for plots and save files
    Xd_pasts, Xa_pasts, Conp_pasts, Cont_pasts, U_pasts, t_pasts = \
    UKF_SNMPC.generate_data(Xd_pasts,Xa_pasts,Conp_pasts,Cont_pasts,U_pasts,un,\
    u_past,xu_pasts,deltat,t_pasts,ws)
           
# Plot results
UKF_SNMPC.plot_graphs(t_past,t_pasts,Xd_pasts,Xa_pasts,U_pasts,Conp_pasts,Cont_pasts)        
        
# Save results
UKF_SNMPC.save_results(Xd_pasts,Xa_pasts,U_pasts,Conp_pasts,Cont_pasts,t_pasts,\
                       xu_pasts,MeanSEx_pasts,CovSEx_pasts)        