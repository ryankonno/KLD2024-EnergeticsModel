'''
Mechanical model

This model is implemented to compute the activation levels of a muscle given a muscle force as input.
Intrinsic properties of the model are based on Dick et al. 2017, and parameters are modified for the
given experiment.

Author: Ryan Konno, University of Queensland
		r.konno@uq.edu.au
'''
################################################################################
# Import
import numpy as np

###############################################################
# Define force-length and force-velocity properties for the muscle
# From Dick et al. 2017
def F_la(e_m, width):
    skew = 0.6
    # width = 0.3
    round = 2.3
    return np.exp(-np.abs((e_m**skew -1)/width)**round)

def F_va(dedt_ce, v_max, a):
    return (1+dedt_ce/v_max)/(1-dedt_ce/v_max/a) * (dedt_ce < 0) \
            + (1.5 - 0.5*(1-dedt_ce/v_max)/(1+7.56 * dedt_ce/v_max / a))*(dedt_ce >0)\
            + (dedt_ce == 0)

# Function for the stiffness of SEE (mainly effect from tendon)
def k_T(F, params):
    # Define parameters
    k_see = params['k_see']
    l_0 = params['l_0']
    F_0 = params['F_0']

    # Caculate the stiffness based on max torque 
    # Beck 2020 or 2022 experiment
    if params['Exp'] == 'Beck2020' or params['Exp'] == 'Beck2022':
        stiffness =  k_see

    # Other experiments (van der Zee 2021)
    else: 
        # Calculate stiffness based on a specified strain value
        max_strain = params['max_strain']
        stiffness = max(F) / max_strain / l_0

    # Exponential stiffness Lichtwark and Wilson 2008
    # Constants from Lichtwark and Wilson 2008
    # Q = 10 # Unitless
    # k_l = 35e3 # N/m
    # stiffness = k_l/F_0 * ( 1 + (0.9/-np.exp(Q * F)))
    # stiffness = max(F)/ max_strain / l_0

    return stiffness

# Compute the rate of change in force
def dFdt(F):
    dFdt = 0 * F # Intialize to correct length
    dFdt[0:-2] = (F[0:-2] - F[1:-1])/dt
    return dFdt

# Calculate the stretch rate using finite differences
def dedt_m(e_m, dt):
    x = e_m 
    x = x*0 
    x[0:-1] = np.diff(e_m)/dt
    return x

# Calculate the stretch of the muscle
# e_m = e_mtu - (l_slack_t + F_t/k_t)/l_slack_t
def e_m(F, params):
    l_0 = params['l_0']
    F_0 = params['F_0']
    s_l = params['s_l']

    return (l_0 - (F / k_T(F,params)) - s_l) / l_0

# Solve the Hill model for the muscle activation, we have
# F_mtu = a * F_l * F_v = F_t
# solving for a gives
# a = (F_mtu)/F_l/F_v
def computeActivation(e_m, dedt_m, F_mtu, v_max, a, width):
    return F_mtu/(F_la(e_m, width)*F_va(dedt_m, v_max, a))

'''
Function to run the mechanical model

Inputs 
	t: s, time
	F: Unitless, Force trace (based on experimental data)
	params: dict of parameters for the model

Outputs
	act: Unitless, Muscle activation at a given time
	e_ce: Unitless, Muscle stretch
	dedt_ce: 1/s, Muscle stretch rate
'''
def runMechModel(t,F,params):
	# Define parameters
    v_max = params['v_max']
    a = params['a']
    width = params['F_la_width']

    dt = t[1] - t[0] # s, Assume constant spacing

    # Compute stretch
    e_ce = e_m(F, params) # Unitless

    # Compute stretch rate
    dedt_ce = dedt_m(e_ce, dt) # s

    # Compute activation
    act = computeActivation(e_ce, dedt_ce, F, v_max, a, width) # Normalized

    return act, e_ce, dedt_ce
