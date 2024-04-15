'''
Code to combine the heat and mechanical model. 

Based on a given force trace, this model will compute the energy rates.

Author: Ryan Konno, University of Queensland
		r.konno@uq.edu.au
'''
################################################################################
# Import
import numpy as np

# Import the heat model
from MuscleModel.HeatModel import dHdt

# Import the mechanical model 
from MuscleModel.MechModel import runMechModel, F_va, F_la

# Import function for scaling energy parameters by fibre type
from MuscleModel.RecruitmentModel import getEnergyParam

################################################################################
# Data structure to hold the model results
class dataStruct:
	def __init__(self):
		self.act_vals=0
		self.max_act = 0 
		self.dHdt_vals = 0
		self.force = 0
		self.dedt_ce = 0
		self.mean_act = 0
		self.e_ce = 0
		self.k_T = 0
		self.F_m = 0
		self.act = 0
		self.alpha_f_r1 = 0
		self.alpha_f_r2 = 0
		self.r1 = 0 
		self.r2 = 0
################################################################################
'''
Function to run the combined heat and mechanical modelj
Inputs 
    - t: s, Time
    - F: unitless, Force trace based on experiment
    - params: Dictionary of parameter values
Output
    - Solution: Data structure
    - Energy Solution: Dictionary of energy results, in W
'''
def runModel(t, F, params):
	# Define parameters
	alpha_f_min = params['alpha_f_min']
	alpha_f_max = params['alpha_f_max']
	r1_min = params['r1_min']
	r2_min = params['r2_min']
	r1_max = params['r1_max']
	r2_max = params['r2_max']
	scale_method = params['scale_method']

	dt = t[1] - t[0] # s, Assume uniform spacing


	# Compute activation from the mechanical model
	# This uses the original methods
	act, e_ce, dedt_ce = runMechModel(t,F,params)

	# Calculate r1 and r2 based on the activation levels 
	if scale_method == 'fibre_act':
		alpha_f_r1, params['r1'] = getEnergyParam(act, r1_min, r1_max, alpha_f_min, alpha_f_max, params)
		alpha_f_r2, params['r2'] = getEnergyParam(act, r2_min, r2_max, alpha_f_min, alpha_f_max, params)
	else: 
		print('Not scaling parameters')
		alpha_f_r1 = 0
		alpha_f_r2 = 0
		params['r1'] = params['r1_min']
		params['r2'] = params['r2_min']


	# Calculate the energetic cost
	dHdt_vals, Energy_data = dHdt(act,t,e_ce,dedt_ce,F,params)

	# Organize the solution to export
	sol = dataStruct()
	sol.act_vals = act
	sol.max_act = max(act)
	sol.dHdt_vals = dHdt_vals
	sol.dedt_ce = dedt_ce
	sol.mean_act = np.mean(act)
	sol.e_ce = e_ce
	sol.F_m = F
	sol.F_la = F_la(e_ce, params['F_la_width'])
	sol.F_va = F_va(dedt_ce, params['v_max'], params['a'])
	sol.act = act
	sol.alpha_f_r1 = alpha_f_r1
	sol.alpha_f_r2 = alpha_f_r2
	sol.r1 = params['r1'] 
	sol.r2 = params['r2']

	return sol, Energy_data