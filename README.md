# Unscented Kalman filter stochastic nonlinear model predictive control
The code in this repository is based on the works in [[1]](#1)[[2]](#2) and is written in Python. For more information on the required modules and packages refer to section [Technical requirements](#Tr). It is a variation of an unscented kalman filter stochastic nonlinear model predictive control (UKF-SNMPC) algorithm. To cite UKF-SNMPC please use the publications [[1]](#1)[[2]](#2). 

## Getting started


## Description
Model predictive control (MPC) is a popular control method, which is however reliant on an accurate dynamic model. Many dynamic systems however are affected by significant uncertainties often leading to a lower performance and significant constraint violations. In this algorithm we assume that a nonlinear system is affected by known stochastic parametric uncertainties leading to a stochastic nonlinear MPC (SNMPC) approach. The square-root Unscented Kalman filter (UKF) equations are used in this context for both estimation and propagation of mean and covariance of the states by generating separate scenarios as shown in the figure above. The uncertainty description is used to optimize an objective in expectation and employ chance-constraints to maintain feasibility despite the presence of the stochastic uncertainties. The covariance of the nonlinear constraints is found using linearization. The dynamic equation system is assumed to be given by differential algebraic equations (DAE). Further description on the theory can be found in [[1]](#1)[[2]](#2). 

<<img src="/images/Image1.jpg" width="500">>

## Features & benefits
* Cheap SNMPC implementation for both receding and shrinking time horizons
* Parameter and state estimation using the square-root UKF
* Nonlinear chance constraints on states to maintain feasibility
* Robust horizon up to which uncertainties are propagated to prevent the growth of open-loop uncertainties
* Efficient solution of nonlinear dynamic optimization formulation using automatic differentiation

## Applications
The algorithm has been shown in [[1]](#1)[[2]](#2) as an efficient tool for the control of a batch process in a shrinking horizon, which is an example of a highly nonlinear and unsteady-state system. These were described by first-principles derived DAEs. See below for example the improved temperature control of the proposed algorithm compared to a nominal NMPC using soft-constraints.

<img src="/images/Image2.jpg" width="800">

## Technical requirements
The code was written using [CasADi](https://web.casadi.org/) in Python 2.7 and hence requires [CasADi](https://web.casadi.org/) with all its sub-dependencies. Simply download a Python 2.7 distribution such as [Python(x,y)](https://python-xy.github.io/) and install CasADi following the [instructions](https://github.com/casadi/casadi/wiki/InstallationInstructions). The HSL linear solvers are also required for IPOPT to work well, see [Obtaining-HSL](https://github.com/casadi/casadi/wiki/Obtaining-HSL) for more information.
<a name="Tr">
</a>

## References
[1] E. Bradford, and L. Imsland, [Stochastic Nonlinear Model Predictive Control with State Estimation by Incorporation of the Unscented Kalman Filter](https://www.researchgate.net/profile/Eric_Bradford/publication/319501430_Stochastic_Nonlinear_Model_Predictive_Control_with_State_Estimation_by_Incorporation_of_the_Unscented_Kalman_Filter/links/59b6774aaca2722453a3a7a9/Stochastic-Nonlinear-Model-Predictive-Control-with-State-Estimation-by-Incorporation-of-the-Unscented-Kalman-Filter.pdf), arXiv preprint arXiv:1709.01201, 2017. 
<a name="1">
</a>

[2] E. Bradford, and L. Imsland, [Economic stochastic model predictive control using the unscented Kalman filter](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2568350/1-s2.0-S2405896318320196-main.pdf?sequence=5), IFAC-PapersOnLine, vol. 51, no. 18, pp. 417-422. 
<a name="2">
</a>

## Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie SklodowskaCurie grant agreement No 675215.

## Legal information
This project is licensed under the MIT license – see LICENSE.md in the repository for details.
