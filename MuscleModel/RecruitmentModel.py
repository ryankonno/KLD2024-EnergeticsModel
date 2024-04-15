'''
Script to calculate the parameters of the energetics model based on the contraction rate and activation level.

This code assumes a recruitment model from slow to fast fibres.

NOTE: This code was not used in Konno et al. 2024

Author: Ryan Konno, University of Queensland
		r.konno@uq.edu.au
'''
###############################################################
# Import
import numpy as np

###############################################################
# Define sinusoidal function to scale from value a_min to a_max 
# taking in an input of x_min to x_max
def sineRamp(x,x_min,x_max,a_min,a_max):
	return 0.5 * (np.sin(np.pi*(x-x_min)/(x_max - x_min) - np.pi / 2) + 1) * (a_max - a_min) + a_min

# Define a linear ramping function 
def linearRamp(x, x_min, x_max, a_min, a_max):
	return (x-x_min)/(x_max - x_min) * (a_max - a_min) + a_min

# Function to obtain the fraction of fast fibres active
def getAlpha_f(act, alpha_f_min, alpha_f_max, params):
	# Define parameters
	act_T = params['act_T']
	act_max = params['act_max']
	return alpha_f_min * (act < act_T) + sineRamp(act,act_T,act_max,alpha_f_min,alpha_f_max) * (act >= act_T) * (act < act_max) + alpha_f_max * (act >= act_max)

###############################################################
# Define function to calculate the energy parameter based on a given activation level
# 	act: activation level
#	 ri_min: minimum value of the energy parameter 
#	 ri_max: maximum value of the energy parameter (with 100% fast fibres)
#	 alpha_f_min: initial fraction of fast fibres activated
#	 alpha_f_max: fraction of fast fibres in the muscle
def getEnergyParam(act, ri_min, ri_max, alpha_f_min, alpha_f_max, params):

	# Get the fraction of fast fibres active
	alpha_f_act =  getAlpha_f(act, alpha_f_min, alpha_f_max, params)
	
	# Sort the parameters into arrays
	alpha_data = np.array((alpha_f_min,0.95))
	ri_data = np.array((ri_min,ri_max))

	# Get the maximum possible energy parameter if all fibres are active (linear interpolation)
	ri_max_shift = np.interp(alpha_f_max,alpha_data,ri_data)

	# Use a linear interpolation to get the energetic parameter 
	ri = linearRamp(alpha_f_act,alpha_f_min,alpha_f_max,ri_min,ri_max_shift)

	return alpha_f_act, ri 