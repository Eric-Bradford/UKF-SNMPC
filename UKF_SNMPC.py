from pylab import *
import numpy as np
from UKF_SNMPC_problem_definition import *
from casadi import *
from scipy.io import savemat
import math as math

class UKF_SNMPC:
    def __init__(self):
        # Variable definitions
        self.xd, self.xa, self.xu, self.u, self.ODEeq, self.Aeq, self.Obj_M,\
        self.Obj_L, self.R, self.u_min, self.u_max, self.states, self.algebraics,\
        self.inputs, self.hfcn, self.Sigma_v, self.ngp, self.gpfcn, self.pgp,\
        self.Sigma_w, self.ngt, self.gtfcn, self.pgt, self.CovP, self.MeanP, self.nm, \
        self.Meanx0, self.Covx0, self.MeanSEx0, self.CovSEx0, self.u0, \
        self.xD_init, self.xA_init, self.u_init = DAE_system()
        self.tf, self.nk, self.shrinking_horizon, self.deg, self.cp, self.nicp,\
        self.simulation_time, self.opts, self.number_of_repeats, self.alpha,\
        self.beta, self.kappa, self.robust_horizon = specifications()
        self.h               = self.tf/self.nk/self.nicp
        self.nd, self.na     = SX.size(self.xd)[0], SX.size(self.xa)[0] 
        self.nu , self.nun   = SX.size(self.u)[0], SX.size(self.xu)[0]
        self.n_L             = self.nd + self.nun
        self.ns, self.deltat = 2*self.n_L+1, self.tf/self.nk
        
        # Internal function calls
        self.C, self.D                                 = self.collocation_points()
        self.ffcn                                      = self.model_fcn()
        self.lambda_ukf, self.nu_m, self.nu_c, self.sqrt_CovP, self.sqrt_Sigma_w\
        , self.sqrt_Sigma_v, self.scaling_factor, self.sqrt_scaling_factor\
                                                       = self.UKF_initialization()
        self.NU, self.NV, self.V, self.vars_lb, self.vars_ub\
        , self.vars_init, self.XD, self.XA, self.U, self.P, self.Sigma_chol\
        , self.x_hat, self.x_hat_a, self.sqrt_Sigma, self.sqrt_Sigma_a\
        , self.phi_before, self.Sigmapoints_a, self.xf_k, self.con\
                                                       = self.NLP_specification()
        self.vars_init, self.vars_lb, self.vars_ub, self.XD, self.XA, self.U\
        , self.x_previous, self.Sigma_previous, self.y_measurement\
        , self.cfcn, self.NE, self.x_hat, self.Sigma_chol, self.p_s \
                                                       = self.set_variable_bounds()
        self.g, self.lbg, self.ubg, self.Sigmapoints_a, self.x_hatfcn, self.Sigmafcn =\
                                        self.set_inequality_constraints_UKF()
        self.g, self.lbg, self.ubg, self.no_dynamic_constraints = \
                                              self.set_inequality_constraints()
        self.g, self.lbg, self.ubg                     =  \
                                              self.set_probability_constraints()
        self.Obj                                       = self.set_objective()                                      
        self.solver                                    = self.create_solver()        
                                      
    def collocation_points(self):
        deg, cp, nk, h = self.deg, self.cp, self.nk, self.h
        C = np.zeros((deg+1,deg+1)) # Coefficients of the collocation equation
        D = np.zeros(deg+1)         # Coefficients of the continuity equation
        
        # All collocation time points
        tau = SX.sym("tau") # Collocation point
        tau_root = [0] + collocation_points(deg,cp)
        T = np.zeros((nk,deg+1))
        for i in range(nk):
            for j in range(deg+1):
                T[i][j] = h*(i + tau_root[j])
        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        for j in range(deg+1):
            L = 1
            for j2 in range(deg+1):
                if j2 != j:
                    L *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])
            lfcn = Function('lfcn', [tau],[L])
        
            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = lfcn(1.0)
            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            tfcn = Function('tfcn', [tau],[tangent(L,tau)])
            for j2 in range(deg+1):
                C[j][j2] = tfcn(tau_root[j2]) 
            
        return C, D    
    
    def model_fcn(self):
        xd, xa, u, xu, ODEeq, Aeq = self.xd, self.xa, self.u, self.xu, self.ODEeq, self.Aeq
        t     =   SX.sym("t")               
        p_s   =   SX.sym("p_s")             
        xddot =   SX.sym("xddot",self.nd)  
        
        res   = []
        for i in range(self.nd):
            res = vertcat(res,ODEeq[i]*p_s - xddot[i]) 
        
        for i in range(self.na):
            res = vertcat(res,Aeq[i])
        
        ffcn = Function('ffcn', [t,xddot,xd,xa,xu,u,p_s],[res])
        
        return ffcn
 
    def UKF_initialization(self):
        alpha, beta, kappa, n_L = self.alpha, self.beta, self.kappa, self.n_L 
        Sigma_w, Sigma_v, CovP  = self.Sigma_w, self.Sigma_v, self.CovP
      
        nu_m = np.zeros(2*n_L+1)
        nu_c = np.zeros(2*n_L+1)
        lambda_ukf = alpha**2*(n_L+kappa)-n_L
        nu_m[0] = (lambda_ukf/(n_L+lambda_ukf))
        nu_c[0] = (lambda_ukf/(n_L+lambda_ukf))+(1-alpha**2+beta)
        for i in range(1,2*n_L+1):
            nu_m[i] = 1./(2*(n_L+lambda_ukf))
            nu_c[i] =  1./(2*(n_L+lambda_ukf)) 
        scaling_factor      = n_L + lambda_ukf
        sqrt_scaling_factor = sqrt(n_L + lambda_ukf)
        
        sqrt_CovP    = chol(CovP)
        sqrt_Sigma_w = chol(Sigma_w)
        sqrt_Sigma_v = chol(Sigma_v)
        
        return lambda_ukf, nu_m, nu_c, sqrt_CovP, sqrt_Sigma_w\
        , sqrt_Sigma_v, scaling_factor, sqrt_scaling_factor    
      
    def NLP_specification(self):
        xd, xa, u, nicp      = self.xd, self.xa, self.u, self.nicp 
        nk, deg, ns, n_L, nm = self.nk, self.deg, self.ns, self.n_L, self.nm
        nd, na, nu, nx       = self.nd, self.na, self.nu, self.nd + self.na  
        
        # Total number of variables
        NXD = nicp*(nk+1)*(deg+1)*nd # Collocated differential states
        NXA = nicp*(nk+1)*deg*na     # Collocated algebraic states
        NU =  nu*nk                  # Parametrized controls
        
        NV = NXD+NXA+NU
        
        # NLP variable vector
        V   = SX.sym("V",NV*ns-NU*(ns-1)+nd*(nk+1)+nd*nd*(nk+1))
        con = SX.sym("con",nu+nd+nd*nd+nm+nk+1)
        
        # All variables with bounds and initial guess
        vars_lb = np.zeros(NV*ns-NU*(ns-1)+nd*(nk+1)+nd*nd*(nk+1))
        vars_ub = np.zeros(NV*ns-NU*(ns-1)+nd*(nk+1)+nd*nd*(nk+1))
        vars_init = np.zeros(NV*ns-NU*(ns-1)+nd*(nk+1)+nd*nd*(nk+1))
        
        # differential states, algebraic states and control matrix definition after
        # discredization
        XD = np.resize(np.array([],dtype=SX),(nk+1,nicp,deg+1,ns)) # NB: same name as above
        XA = np.resize(np.array([],dtype=SX),(nk+1,nicp,deg,ns)) # NB: same name as above
        U = np.resize(np.array([],dtype=SX),nk+1)
        P = np.resize(np.array([],dtype=SX),ns)
        Sigma_chol = np.resize(np.array([],dtype=SX),(nk+1,nd,nd))
        x_hat = np.resize(np.array([],dtype=SX),(nk+1,nd))
        x_hat_a = np.resize(np.array([],dtype=SX),(nk+1))
        sqrt_Sigma = np.resize(np.array([],dtype=SX),(nk+1))
        sqrt_Sigma_a = np.resize(np.array([],dtype=SX),(nk+1))
        phi_before = np.resize(np.array([],dtype=SX),2*n_L+1)
        Sigmapoints_a = np.resize(np.array([],dtype=SX),(nk+1,ns))
        xf_k = np.resize(np.array([],dtype=SX),(nk+1))
        
        return NU, NV, V, vars_lb, vars_ub, vars_init, XD, XA, U, P, Sigma_chol,\
        x_hat, x_hat_a, sqrt_Sigma, sqrt_Sigma_a, phi_before, Sigmapoints_a,\
        xf_k, con

    def set_variable_bounds(self):
        V, nk, nicp, deg        = self.V, self.nk, self.nicp, self.deg
        deg, nd, XA, na, XD     = self.deg, self.nd, self.XA, self.na, self.XD
        vars_lb, vars_ub, vars_init = self.vars_lb, self.vars_ub, self.vars_init
        nu, U                   = self.nu, self.U
        u_min, u_max            = self.u_min, self.u_max
        ns, NV, NU              = self.ns, self.NV, self.NU
        nm, x_hat               = self.nm, self.x_hat
        Sigma_chol, con         = self.Sigma_chol, self.con
        nx                      = nd + na
        
        u_init  = self.u_init
        xD_init = self.xD_init
        xA_init = self.xA_init
        xD_min  = np.array([-inf]*nd)
        xD_max  = np.array([inf]*nd)
        xA_min  = np.array([-inf]*na)
        xA_max  = np.array([inf]*na)
        
        # Extra variables
        NE = NU 
        
        # Auxiliary variables
        vars_lba   = np.zeros(NV-NE)
        vars_uba   = np.zeros(NV-NE)
        vars_inita = np.zeros(NV-NE)
        
        for l in range(ns):
            offset = 0            
            # Get collocated states and parametrized control
            for k in range(nk+1):  
                # Collocated states
                for i in range(nicp):
                    for j in range(deg+1):
                        # Get the expression for the state vector
                        XD[k][i][j][l] = V[offset+(NV-NE)*l:offset+nd+(NV-NE)*l]
                        if j !=0:
                            XA[k][i][j-1][l] = V[offset+nd+(NV-NE)*l:offset+nd+na+(NV-NE)*l]
                        # Add the initial condition
                        index = (deg+1)*(nicp*k+i) + j
                        
                        if k==0 and j==0 and i==0:
                            vars_inita[offset:offset+nd] = xD_init
                            
                            vars_lba[offset:offset+nd] = xD_min
                            vars_uba[offset:offset+nd] = xD_max                    
                            offset += nd
                        else:
                            if j!=0:
                                vars_inita[offset:offset+nx] = np.append(xD_init,xA_init) 
                                
                                vars_lba[offset:offset+nx] = np.append(xD_min,xA_min)
                                vars_uba[offset:offset+nx] = np.append(xD_max,xA_max)
                                offset += nx
                            else:
                                vars_inita[offset:offset+nd] = xD_init
                                
                                vars_lba[offset:offset+nd] = xD_min
                                vars_uba[offset:offset+nd] = xD_max
                                offset += nd
                        
            assert(offset==NV-NE)
      
            # Set variable bounds and initial guess for scenarious
            vars_lb[(NV-NE)*l:(NV-NE)*(l+1)] = vars_lba
            vars_ub[(NV-NE)*l:(NV-NE)*(l+1)] = vars_uba
            vars_init[(NV-NE)*l:(NV-NE)*(l+1)] = vars_inita
        
        # Set control contraints
        for i in range(nk):
            if i == 0:
                U[i+1] = V[(NV-NE)*(l+1)+NU-nu:(NV-NE)*(l+1)+NU]
                vars_lb[(NV-NE)*(l+1)+NU-nu:(NV-NE)*(l+1)+NU]   = u_min
                vars_ub[(NV-NE)*(l+1)+NU-nu:(NV-NE)*(l+1)+NU]   = u_max
                vars_init[(NV-NE)*(l+1)+NU-nu:(NV-NE)*(l+1)+NU] = u_init
            else:
                vars_lb[(NV-NE)*(l+1)+NU-(i+1)*nu:(NV-NE)*(l+1)+NU-i*nu]   = u_min
                vars_ub[(NV-NE)*(l+1)+NU-(i+1)*nu:(NV-NE)*(l+1)+NU-i*nu]   = u_max
                vars_init[(NV-NE)*(l+1)+NU-(i+1)*nu:(NV-NE)*(l+1)+NU-i*nu] = u_init
                U[i+1] = V[(NV-NE)*(l+1)+NU-(i+1)*nu:(NV-NE)*(l+1)+NU-i*nu]
    
        # Set x_previous, Sigma_previous, U_previous and y_measurement
        U[0]           = con[:nu] 
        x_previous     = con[nu:nu+nd]
        Sigma_previous = con[nu+nd:nu+nd+nd*nd].reshape((nd,nd))
        y_measurement  = con[nu+nd+nd*nd:nu+nd+nd*nd+nm]
        p_s            = con[nu+nd+nd*nd+nm:nu+nd+nd*nd+nm+nk+1]
      
        variable_offset = (NV-NE)*ns+NU-1
        for i in range(nk+1):
            for j in range(nd):
                variable_offset += 1
                x_hat[i][j] = V[variable_offset]
                vars_init[variable_offset] = xD_init[j]
                vars_lb[variable_offset]   = -inf*np.ones(1)
                vars_ub[variable_offset]   =  inf*np.ones(1)
        
        initial_guess = chol(np.eye(nd)*0.01)       
        for i in range(nk+1):
            for j in range(nd):
                for k in range(nd):
                    variable_offset += 1
                    Sigma_chol[i][j][k]        = V[variable_offset]
                    vars_init[variable_offset] = initial_guess[j,k]
                    vars_lb[variable_offset]   = -inf*np.ones(1)
                    vars_ub[variable_offset]   =  inf*np.ones(1)
            
        cfcn = Function('cfcn',[V],[U[1]])    
        
        return vars_init, vars_lb, vars_ub, XD, XA, U, x_previous,\
        Sigma_previous, y_measurement, cfcn, NE, x_hat, Sigma_chol, p_s

    def set_inequality_constraints(self):
        V, nk, nicp, deg        = self.V, self.nk, self.nicp, self.deg
        XD, nd, XA, na          = self.XD, self.nd, self.XA, self.na
        ffcn                    = self.ffcn
        Sigmapoints_a           = self.Sigmapoints_a
        p_s, ns, C, h, U        = self.p_s, self.ns, self.C, self.h, self.U
        g, lbg, ubg             = self.g, self.lbg, self.ubg
        
        t   = SX.sym('t')
        
        for k in range(nk+1):
            for l in range(ns):
                    
                # For all finite elements
            
                # Set uncertain parameters
                XU = Sigmapoints_a[k][l][nd:]
                
                for i in range(nicp):
                    # For all collocation points
                    for j in range(1,deg+1):                
                        # Get an expression for the state derivative at the collocation point
                        xp_jk = 0
                        for j2 in range (deg+1):
                            xp_jk += C[j2][j]*XD[k][i][j2][l]       # get the time derivative of the differential states (eq 10.19b) 
                        # Add collocation equations to the NLP
                        fk = ffcn(0., xp_jk/h, XD[k][i][j][l], XA[k][i][j-1][l], XU, U[k], p_s[k])
                        g += [fk[:nd]]                     # impose system dynamics (for the differential states (eq 10.19b))
                        lbg.append(np.zeros(nd)) # equality constraints
                        ubg.append(np.zeros(nd)) # equality constraints
                        g += [fk[nd:]]                               # impose system dynamics (for the algebraic states (eq 10.19b))
                        lbg.append(np.zeros(na)) # equality constraints
                        ubg.append(np.zeros(na)) # equality constraints
                    
                    # Get an expression for the state at the end of the finite element
                    # Add continuity equation to NLP
                    if i != nicp-1:
                        g += [XD[k][i+1][0][l] - XD[k][i][deg][l]]
                        lbg.append(np.zeros(nd))
                        ubg.append(np.zeros(nd))  
                
            # Number of dyanmic/continuity constraints
            no_dynamic_constraints = len(np.concatenate(lbg))
            
        return g, lbg, ubg, no_dynamic_constraints
         
    def cholupdate(self,R1,x1,sign1):
      p1 = SX.size(x1)[0]
      x1 = transpose(x1)
      for k in range(p1):
        if sign1 == '+':
          r1 = sqrt(R1[k,k]**2 + x1[k]**2)
        elif sign1 == '-':
          r1 = sqrt(R1[k,k]**2 - x1[k]**2)
        c = r1/R1[k,k]
        s = x1[k]/R1[k,k]
        R1[k,k] = r1
        if k+1 < p1:
            if sign1 == '+':
              R1[k,k+1:p1] = (R1[k,k+1:p1] + s*x1[k+1:p1])/c
            elif sign1 == '-':
              R1[k,k+1:p1] = (R1[k,k+1:p1] - s*x1[k+1:p1])/c
            x1[k+1:p1]= c*x1[k+1:p1] - s*R1[k,k+1:p1]
            
      return R1
    
    def Unscented_transformation(self,nu_m,nu_c,nd,sqrt_Sigma_w,ns,XD,k,nicp,deg,U):
        Transformed_sampling  = SX.zeros(nd,ns)
        Sum_mean              = SX.zeros(nd,1)
        cholupdate            = self.cholupdate
    
        for i in range(ns):
                Transformed_sampling[:,i] = XD[k][nicp-1][deg][i]
                Sum_mean += nu_m[i]*Transformed_sampling[:,i]
        
        Sum_mean_matrix = []
        for i in range(ns):
            Sum_mean_matrix = horzcat(Sum_mean_matrix,Sum_mean)   
        Sum_mean_matrix1 = Transformed_sampling - Sum_mean_matrix 
        residual = mtimes(Sum_mean_matrix1,diag(sqrt(fabs(nu_c))))
        
        Aux = transpose(horzcat(residual[:,1:ns],sqrt_Sigma_w))
        Sigma_prediction_a = qr(Aux)[1]
        
        if nu_c[0] < 0:
            Sigma_prediction = cholupdate(Sigma_prediction_a,residual[:,0],'-')
        else:
            Sigma_prediction = cholupdate(Sigma_prediction_a,residual[:,0],'+')
            
        return Sum_mean, Transformed_sampling, Sigma_prediction, Sum_mean_matrix1
    
    def Unscented_transformation_m(self,nu_m,nu_c,nm,sqrt_Sigma_v,ns,XD,k,nicp,deg,hfcn):
        nd                     = self.nd
        Sum_mean_m             = SX.zeros(nm,1)
        Transformed_sampling_m = SX.zeros(nm,ns)
        cholupdate             = self.cholupdate
        
        for i in range(ns):
                XU                          = self.Sigmapoints_a[k][i][nd:]
                Transformed_sampling_m[:,i] = hfcn(XD[k][nicp-1][deg][i],XU) 
                Sum_mean_m                 += nu_m[i]*Transformed_sampling_m[:,i]
        x_mean_m = Sum_mean_m
        
        Sum_mean_matrix_m  = []
        for i in range(ns):
            Sum_mean_matrix_m = horzcat(Sum_mean_matrix_m,Sum_mean_m)   
        Sum_mean1_matrix_m = Transformed_sampling_m - Sum_mean_matrix_m 
        residual_m = mtimes(Sum_mean1_matrix_m,diag(sqrt(fabs(nu_c))))
        
        Aux = transpose(horzcat(residual_m[:,1:ns],sqrt_Sigma_v))
        Sigma_prediction_a_m = qr(Aux)[1]
        Sigma_prediction_a_m = Sigma_prediction_a_m[:nm,:]
        
        if nu_c[0] < 0:
            Sigma_prediction_m = cholupdate(Sigma_prediction_a_m,residual_m[:,0],'-')
        else:
            Sigma_prediction_m = cholupdate(Sigma_prediction_a_m,residual_m[:,0],'+')
        
        return x_mean_m, Transformed_sampling_m, Sigma_prediction_m, Sum_mean1_matrix_m 

    def Sigma_points(self,x_hat_before,Sigma_before_chol,nun,sqrt_CovP,MeanP,nd,lambda_ukf,n_L):
        ns               = self.ns
        x_hat_a_previous = vertcat(x_hat_before,MeanP)
        
        sqrt_Sigma_previous         = Sigma_before_chol
        horizontal_stack_sqrt_Sigma = horzcat(sqrt_Sigma_previous,SX.zeros(nd,nun))
        horizontal_stack_sqrt_P     = horzcat(SX.zeros(nun,nd),sqrt_CovP)
        sqrt_Sigma_a_previous = vertcat(horizontal_stack_sqrt_Sigma,horizontal_stack_sqrt_P)
          
        x_hat_vector = SX.zeros((n_L,(ns-1)//2))  
        for i in range((ns-1)//2):
          x_hat_vector[:,i] =  x_hat_a_previous
          
        Sigmapoint_matrix = sqrt(n_L+lambda_ukf)*sqrt_Sigma_a_previous
        Sigmapoints_aug = horzcat(x_hat_a_previous,x_hat_vector + Sigmapoint_matrix, \
        x_hat_vector - Sigmapoint_matrix) 
        Sigmapoints = np.resize(np.array([],dtype=SX),ns)
        
        for l in range(2*n_L+1):
            Sigmapoints[l] = Sigmapoints_aug[:,l]
            
        return Sigmapoints

    def set_inequality_constraints_UKF(self):
        nk, Sigma_chol, x_previous    = self.nk, self.Sigma_chol, self.x_previous
        nun, nm                       = self.nun , self.nm  
        sqrt_CovP, MeanP              = self.sqrt_CovP, self.MeanP
        sqrt_Sigma_w, sqrt_Sigma_v    = self.sqrt_Sigma_w, self.sqrt_Sigma_v
        lambda_ukf                    = self.lambda_ukf
        nu_m, nu_c, n_L, nd           = self.nu_m, self.nu_c, self.n_L, self.nd
        deg, nicp, hfcn, XD           = self.deg, self.nicp, self.hfcn, self.XD
        U, phi_before, ns             = self.U, self.phi_before, self.ns
        Sigma_w, Sigma_v              = self.Sigma_w, self.Sigma_v   
        y_measurement, scaling_factor = self.y_measurement, self.scaling_factor
        sqrt_scaling_factor           = self.sqrt_scaling_factor
        robust_horizon                = self.robust_horizon
        cholupdate                    = self.cholupdate
        Unscented_transformation      = self.Unscented_transformation
        Unscented_transformation_m    = self.Unscented_transformation_m
        Sigma_points                  = self.Sigma_points
        Sigma_previous, Sigmapoints_a = self.Sigma_previous, self.Sigmapoints_a 
        x_hat, Sigma_chol, V          = self.x_hat, self.Sigma_chol, self.V
        
        # Constraint function for the NLP
        g = []
        lbg = []
        ubg = []
        k = 0
    
        Sigmapoints_a[k] = \
        Sigma_points(x_previous,Sigma_previous,nun,sqrt_CovP,MeanP,nd,lambda_ukf,n_L)
            
        for l in range(2*n_L+1):
            for i in range(nd):
                g += [XD[k][0][0][l][i] - Sigmapoints_a[k][l][i]]
                lbg.append(np.zeros(1))
                ubg.append(np.zeros(1))
            
        # Prediction
        x_hat_before,Transformed_sampling_pointsp,Sigma_before,\
        Transformed_deviationsp = Unscented_transformation(nu_m,nu_c,nd,sqrt_Sigma_w, \
                                ns,XD,k,nicp,deg,U)
        
        # Observation transformation
        y_hat_before,Transformed_sampling_pointsm, Sigma_y_y,\
        Transformed_deviationsm = Unscented_transformation_m(
        nu_m,nu_c,nm,sqrt_Sigma_v,ns,XD,k,nicp,deg,hfcn)        
        
        Sigma_x_y = mtimes(mtimes(Transformed_deviationsp,diag(nu_c)),transpose(Transformed_deviationsm))
        invSigma_y_y = solve(Sigma_y_y,SX.eye(nm))
        
        # Measurement update
        K = mtimes(mtimes(Sigma_x_y,invSigma_y_y),transpose(invSigma_y_y))
        U1 = mtimes(K,transpose(Sigma_y_y))
        S1 = Sigma_before
        for i in range(nm):
            S1 = cholupdate(S1,U1[:,i],'-')
        S = S1
        
        vector_x_hat = (x_hat_before + mtimes(K,(y_measurement-y_hat_before)))
        for i in range(nd):
            g += [x_hat[k][i] - vector_x_hat[i]] 
            lbg.append(np.zeros(1))
            ubg.append(np.zeros(1))   
         
        for i in range(nd):
           for j in range(nd):
            g += [Sigma_chol[k][i][j] - S[i,j]] 
            lbg.append(np.zeros(1))
            ubg.append(np.zeros(1)) 
        
        for k in range(1,nk+1):
            Sigmapoints_a[k] = Sigma_points(x_hat[k-1],Sigma_chol[k-1],nun,sqrt_CovP,MeanP,nd,lambda_ukf,n_L) 
            
            for l in range(2*n_L+1):
                for i in range(nd):
                    g += [XD[k][0][0][l][i] - Sigmapoints_a[k][l][i]]
                    lbg.append(np.zeros(1))
                    ubg.append(np.zeros(1))
                        
            # Prediction
            vector_x_hat,Transformed_sampling_pointsp,Sigma_vector,\
            Transformed_deviationsp = Unscented_transformation(nu_m,nu_c,nd,sqrt_Sigma_w \
            ,ns,XD,k,nicp,deg,U)             
             
            for i in range(nd):
                g += [x_hat[k][i] - vector_x_hat[i]] 
                lbg.append(np.zeros(1))
                ubg.append(np.zeros(1))              
            
            if k <= robust_horizon:
                for i in range(nd):
                   for j in range(nd):
                        g += [Sigma_chol[k][i][j] - Sigma_vector[i,j]] 
                        lbg.append(np.zeros(1))
                        ubg.append(np.zeros(1)) 
            else:
                for i in range(nd):
                   for j in range(nd):
                        g += [Sigma_chol[k][i][j] - Sigma_chol[k-1][i][j]] 
                        lbg.append(np.zeros(1))
                        ubg.append(np.zeros(1)) 
            
            Sigma_data = np.resize(np.array([],dtype=SX),(nk+1,nd,nd))
            
            for k in range(nk+1):
                Sigma_true = mtimes(transpose(Sigma_chol[k]),Sigma_chol[k])
                for i in range(nd):
                    for j in range(nd):
                        Sigma_data[k][i][j] = Sigma_true[i,j]
        
        x_hatfcn = Function('x_hatfcn',[V],[x_hat[0]])
        Sigmafcn = Function('Sigmafcn',[V],[mtimes(Sigma_chol[0].T,Sigma_chol[0])])
        
        return g, lbg, ubg, Sigmapoints_a, x_hatfcn, Sigmafcn 
    
    def set_probability_constraints(self):   
        Sigma_chol, xd, nk           = self.Sigma_chol, self.xd, self.nk
        x_hat, g, lbg, ubg           = self.x_hat, self.g, self.lbg, self.ubg
        ngp, gpfcn, pgp              = self.ngp, self.gpfcn, self.pgp
        ngt, gtfcn, pgt              = self.ngt, self.gtfcn, self.pgt
        
        for k in range(1,nk+1):
            for i in range(ngp):
                ke        = sqrt((1-pgp[i])/pgp[i])
                Sigma_cov = mtimes(transpose(Sigma_chol[k]),Sigma_chol[k])
                meangp    = gpfcn(x_hat[k])[i] 
                Jgp       = jacobian(gpfcn(x_hat[k])[i],x_hat[k])
                vargp     = mtimes(mtimes(Jgp,Sigma_cov),Jgp.T) 
                g        += [meangp + sqrt(vargp+1e-8)*ke]
                lbg.append([-inf])
                ubg.append([0.])
        
        for i in range(ngt):
            ke        = sqrt((1-pgt[i])/pgt[i])
            Sigma_cov = mtimes(transpose(Sigma_chol[nk]),Sigma_chol[nk])
            meangt    = gtfcn(x_hat[nk])[i]
            Jgt       = jacobian(gtfcn(x_hat[nk])[i],x_hat[nk])
            vargt     = mtimes(mtimes(Jgt,Sigma_cov),Jgt.T) 
            g        += [meangt + sqrt(vargt+1e-8)*ke]
            lbg.append([-inf])
            ubg.append([0.])
        
        return g, lbg, ubg
    
    def set_objective(self):
        x_hat, Sigma_chol, nk = self.x_hat, self.Sigma_chol, self.nk
        ns, XD, nd, p_s, R    = self.ns, self.XD, self.nd, self.p_s, self.R
        Obj_M, Obj_L , U      = self.Obj_M, self.Obj_L, self.U
        
        delta_U          = SX.zeros(1)
        ObjL             = SX.zeros(1)
        Sigma_cov        = mtimes(transpose(Sigma_chol[nk]),Sigma_chol[nk])
        for k in range(nk):
                delta_U += mtimes(mtimes(transpose(U[k+1]-U[k]),R),U[k+1]-U[k])*p_s[k]
        for k in range(1,nk+1):
            ObjL        += Obj_L(x_hat[k],U[k])*p_s[k]
        
        Obj              = delta_U + Obj_M(x_hat[-1],U[-1]) + ObjL
        
        return Obj
    
    def create_solver(self):
       V, con, Obj, g, opts = self.V, self.con, self.Obj, self.g, self.opts
       
       # Define NLP
       nlp = {'x':V, 'p':con, 'f':Obj, 'g':vertcat(*g)} 
            
       # Allocate an NLP solver
       solver = nlpsol("solver", "ipopt", nlp, opts)   
       
       return  solver 
     
    def simulator(self,xd_previous,uNMPC,t0,tf,xu_real):
        xd, xa, u, ODEeq, Aeq = self.xd, self.xa, self.u, self.ODEeq, self.Aeq
        xu = self.xu

        ODE = []
        for i in range(self.nd):
            ODE = vertcat(ODE,substitute(ODEeq[i],vertcat(u,xu),vertcat(SX(uNMPC),SX(xu_real)))) 
        
        A = []
        for i in range(self.na):
            A   = vertcat(A,substitute(Aeq[i],vertcat(u,xu),vertcat(SX(uNMPC),SX(xu_real))))
        
        dae = {'x':xd, 'z':xa, 'ode':ODE, 'alg':A}        
        I = integrator('I', 'idas', dae, {'t0':t0, 'tf':tf, 'abstol':1e-10, \
        'reltol':1e-10})
        res        = I(x0=xd_previous)
        xd_current = np.array(res['xf'])
        xa_current = np.array(res['zf'])
        
        return xd_current, xa_current 
     
    def initialization(self):
        tf, deltat, nu, nd         = self.tf, self.deltat, self.nu, self.nd
        number_of_repeats, na, ngp = self.number_of_repeats, self.na, self.ngp
        nun, nk, ngt               = self.nun, self.nk, self.ngt

        U_pasts       = np.zeros((number_of_repeats,int(math.ceil(tf/deltat)),nu))
        Xd_pasts      = np.zeros((int(math.ceil(tf/deltat))*100+1,number_of_repeats,nd))
        Xa_pasts      = np.zeros((int(math.ceil(tf/deltat))*100,number_of_repeats,na)) 
        Conp_pasts    = np.zeros((int(math.ceil(tf/deltat))*100,number_of_repeats,ngp))
        Cont_pasts    = np.zeros((int(math.ceil(tf/deltat))*100,number_of_repeats,ngt))
        xu_pasts      = np.zeros((number_of_repeats,nun)) 
        MeanSEx_pasts = np.zeros((nk+1,number_of_repeats,nd))
        CovSEx_pasts  = np.zeros((nk+1,number_of_repeats,nd,nd))
        t_pasts       = np.zeros((int(math.ceil(tf/deltat))*100+1,number_of_repeats))

        return U_pasts, Xd_pasts, Xa_pasts, Conp_pasts, Cont_pasts, xu_pasts, MeanSEx_pasts,\
        CovSEx_pasts, t_pasts
    
    def initialization_loop(self):
        Meanx0, Covx0 = self.Meanx0, self.Covx0
        lbg, ubg      = self.lbg, self.ubg
        vars_lb, vars_ub, vars_init = self.vars_lb, self.vars_ub, self.vars_init
        tf, deltat, nu, nd = self.tf, self.tf/self.nk, self.nu, self.nd
        number_of_repeats, na  = self.number_of_repeats, self.na
        MeanSEx, CovSEx = self.MeanSEx0, self.CovSEx0    
    
        arg = {} 
        arg["lbg"] = np.concatenate(lbg)
        arg["ubg"] = np.concatenate(ubg)
        arg["lbx"] = vars_lb
        arg["ubx"] = vars_ub
        arg["x0"]  = vars_init        
        
        u_nmpc    = np.array(self.u0)
        u_past    = []
        t_past    = [0.]
        tk        = -1
        t0i       = np.array([[0.]])
        tfi       = t0i + deltat
        
        return arg, u_past, tk, t0i, tfi, u_nmpc, Meanx0, Covx0, MeanSEx, CovSEx, t_past  
    
    def update_inputs(self,yd,tk,u_nmpc,MeanSEx,CovSEx):
        nu, nd, nm, nk    = self.nu, self.nd, self.nm, self.nk
        shrinking_horizon = self.shrinking_horizon
        
        tk += 1
        p   = np.zeros(nu+nd+nd*nd+nm+nk+1)
        
        if shrinking_horizon:
            a = np.concatenate((np.ones(nk+1-tk),np.zeros(tk)))
        else:
            a = np.ones(nk+1)
        
        p[:nu]                                = np.array(u_nmpc).flatten()
        p[nu:nu+nd]                           = np.array(MeanSEx).flatten()
        p[nu+nd:nu+nd+nd*nd]                  = np.array(np.reshape(CovSEx,(nd*nd))).flatten()
        p[nu+nd+nd*nd:nu+nd+nd*nd+nm]         = np.array(yd).flatten() 
        p[nu+nd+nd*nd+nm:nu+nd+nd*nd+nm+nk+1] = a
        
        return p, tk
    
    def collect_data(self,t_past,u_past,t0i,u_nmpc):
        t_past += [t0i] 
        u_past += [u_nmpc]
        
        return t_past, u_past
    
    def generate_data(self,Xd_pasts,Xa_pasts,Conp_pasts,Cont_pasts,U_pasts,un,\
                      u_past,xu_pasts,deltat,t_pasts,ws):
        simulation_time  = self.simulation_time
        t_pasts[0,un]    = 0.
        xds              = Xd_pasts[0,un,:]
        t0is             = 0. # start time of integrator
        tfis             = 0. # end time of integrator
        l = 0
        nu, nk, nd       = self.nu, self.nk, self.nd
        
        for k in range(nk):
                for i in range(nu):
                    U_pasts[un][k][i] = u_past[k][i]
        
        for k in range(nk):
            for o in range(100):
                l += 1
                tfis += deltat/100
                xds, xas = self.simulator(xds,u_past[k],t0is,tfis,xu_pasts[un,:])
                Xd_pasts[l,un,:]     = xds[:,0]
                Xa_pasts[l-1,un,:]   = xas[:,0]
                Conp_pasts[l-1,un,:] = np.array(DM(self.gpfcn(xds))).flatten()
                Cont_pasts[l-1,un,:] = np.array(DM(self.gtfcn(xds))).flatten()
                t0is += deltat/100
                t_pasts[l,un]       = t0is 
            xds = xds.flatten() + ws[k][:nd]
        
        return Xd_pasts, Xa_pasts, Conp_pasts, Cont_pasts, U_pasts, t_pasts
    
    def plot_graphs(self,t_past,t_pasts,Xd_pasts,Xa_pasts,U_pasts,Conp_pasts,Cont_pasts):
        states               = self.states
        algebraics           = self.algebraics
        inputs               = self.inputs
        number_of_repeats    = self.number_of_repeats
        nd, na, nu, ngp, ngt = self.nd, self.na, self.nu, self.ngp ,self.ngt
        for j in range(nd):
            plt.figure(j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[:,i],Xd_pasts[:,i,j],'-')
            plt.ylabel(states[j])
            plt.xlabel('time')
            plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
     
        for j in range(na):
            plt.figure(nd+j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[1:,i],Xa_pasts[:,i,j],'-')
            plt.ylabel(algebraics[j])
            plt.xlabel('time')
            plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
        
        for k in range(nu):
            plt.figure(nd+na+k)
            t_pastp = np.sort(np.concatenate([list(xrange(self.nk+1))]*2))
            plt.clf()
            for j in range(number_of_repeats):
                u_pastpF = []
                for i in range(len(U_pasts[j])):
                    u_pastpF += [U_pasts[j][i][k]]*2
                plt.plot(t_pastp[1:-1],u_pastpF,'-')
            plt.ylabel(inputs[k])
            plt.xlabel('time')
            plt.xlim([0,self.nk])  
        
        for j in range(ngp):
            plt.figure(nd+na+nu+j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[1:,i],Conp_pasts[:,i,j],'-')
            plt.ylabel('gp'+str(j))
            plt.xlabel('time')
            plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
            
        for j in range(ngt):
            plt.figure(nd+na+nu+ngt+j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[1:,i],Cont_pasts[:,i,j],'-')
            plt.ylabel('gt'+str(j))
            plt.xlabel('time')
            plt.xlim([0,np.ndarray.max(t_pasts[-1,:])])
        
        return
    
    def save_results(self,Xd_pasts,Xa_pasts,U_pasts,Conp_pasts,Cont_pasts,t_pasts\
                     ,xu_pasts,MeanSEx_pasts,CovSEx_pasts):
        
        Data_UKF_SNMPC                                = {}
        Data_UKF_SNMPC['differential_states']         = Xd_pasts
        Data_UKF_SNMPC['algebraic_states']            = Xa_pasts
        Data_UKF_SNMPC['inputs']                      = U_pasts
        Data_UKF_SNMPC['path_constraints']            = Conp_pasts
        Data_UKF_SNMPC['end_constraints']             = Cont_pasts
        Data_UKF_SNMPC['simulation_times']            = t_pasts
        Data_UKF_SNMPC['uncertain_parameters']        = xu_pasts
        Data_UKF_SNMPC['state_estimates_means']       = MeanSEx_pasts
        Data_UKF_SNMPC['state_estimates_covariances'] = CovSEx_pasts
        savemat('Data_UKF_SNMPC',Data_UKF_SNMPC)
        
        return