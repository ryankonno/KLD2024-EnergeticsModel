'''
Heat model based on Lichtwark and Wilson 2005. 

Heat rates are computed first using unitless quantities, then scaled to W.

Author: Ryan Konno, University of Queensland
		r.konno@uq.edu.au
'''
################################################################################
# Import
import numpy as np

# Import the force-length relationship for the mechanical model 
from MuscleModel.MechModel import F_la

################################################################################
def dHdt(act, t, e_ce, dedt_ce, F, params):
	# Define parameters
	l_0 = params['l_0']
	F_0 = params['F_0']
	r1 = params['r1']
	r2 = params['r2']
	width = params['F_la_width']
	mass = params['mass']

	dt = t[1]-t[0] # s, Assume constant spacing

	# Compute the total maintenance heat rate scaled by activation level
	def dQmdt_total(t, act, e_m, dedt_m, F):
		
		# Maintenance heat rate
		def dQmdt(t, act, e_m, dedt_m, F):
			v_ce_g0 = 0.3 + 0.7 * np.exp(-8 * dedt_m) # Adjusted for +ive dedt_m during lengthening
			return r1 * ((dedt_m < 0) + v_ce_g0 * (dedt_m >= 0))

		return act * dQmdt(t, act, e_m, dedt_m, F) * (0.3 + 0.7 * F_la(e_m, width))

	# Compute the total shortening lengthening heat rate scaled by activation level
	def dQsldt_total(t, act, e_m, dedt_m, F):

		# Shortening and lengthening heat rate
		def dQsldt(act, e_m, dedt_ce, F):
			# return - r2 * dedt_ce * (dedt_ce < 0) + -0.4 * F * (-dedt_ce) * (dedt_ce >=0) # Assume 40% of work is lost to heat
			return - r2 * dedt_ce * (dedt_ce < 0) + - F * (-dedt_ce) * (dedt_ce >=0) # Assume all work is lost to heat

		return act * F_la(e_m, width) * dQsldt(act, e_m, dedt_m, F)

	# Compute the total heat rate
	def dQdt(t, act, e_m, dedt_m, F):
		return dQmdt_total(t, act, e_m, dedt_m, F) + dQsldt_total(t, act, e_m, dedt_m, F)

	# Calculate the work done by the contractile unit
	def W(dedt_ce, F):
		return -dedt_ce * F # Output in 1/s

	# Compute the energy rates and scale to W

	dEinitdt = (dQdt(t, act, e_ce, dedt_ce, F) + W(dedt_ce, F)) * F_0 * l_0
	Q_rec = params['recovery_ratio'] * dEinitdt
	dEdt = (dEinitdt + Q_rec * params['include_recovery'])  # W
	Q_tot    = dQdt(t, act, e_ce, dedt_ce, F) * F_0 * l_0
	Q_m_tot  = dQmdt_total(t, act, e_ce, dedt_ce, F) * F_0 * l_0
	Q_sl_tot = dQsldt_total(t, act, e_ce, dedt_ce, F) * F_0 * l_0
	W_tot    = W(dedt_ce,F) * F_0 * l_0

	# Integrate to get the energy in J
	Sum_QM   = np.cumsum(Q_m_tot * dt)
	Sum_QSL  = np.cumsum(Q_sl_tot * dt)
	Sum_Qtot = np.cumsum(Q_tot * dt)
	Sum_Wtot = np.cumsum(W_tot * dt)
	Sum_E    = np.cumsum(dEdt * dt)
	Sum_Qrec = np.cumsum(Q_rec * dt) 

	# Store data
	Energy_data = {
	    'Q_m': Sum_QM,
	    'Q_sl': Sum_QSL,
	    'Q_tot': Sum_Qtot,
	    'W_tot': Sum_Wtot,
		'Q_rec': Sum_Qrec,
	    'E': Sum_E
	}

	return dEdt, Energy_data