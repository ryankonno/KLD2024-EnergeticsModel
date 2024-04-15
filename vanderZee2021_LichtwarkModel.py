'''
Code to compute energy rates based on torque traces from van der Zee and Kuo (2021).

Author: Ryan Konno, University of Queensland
        r.konno@uq.edu.au
'''
###############################################################
# Import statements
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
font = {'size'   : 8}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = [9.0/2.56, 8.0/2.56]
plt.rcParams['figure.dpi'] = 150
plt.rcParams["svg.fonttype"] = "none"
from matplotlib.ticker import StrMethodFormatter

# Import the combined mech heat model 
from MuscleModel.MuscleModel import runModel

###############################################################
# Define plotting data (color and marker list)
palette = ['#066b0d', '#447a0a', '#838907', '#c19703', '#ffa600']
markerlist = ('solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5, 1, 5)))

###############################################################
# Set the parameter values
params = {
    'Exp': 'vZ2021',                               # String, Experiment name used to specify model specific options in muscle model

    'l_0': 0.095,                                  # double, Muscle fasicle length, m, van der Zee & Kou 2021
    'F_0': 4775,                                   # double, Maximum isometric force, N, Quad, Handsfield et al. 2014

    'v_max': 5,                                    # double, Maximum shortening rate, 1/s, Dick et al. 2017
    'a': 0.17,                                     # double, Force-velocity relationship curvature, unitless, Dick et al. 2017

    'mass': (110.6 + 375.9 + 171.9 + 239.4),       # double, Muscle mass g, (Ward et al. 2009, Total quad mass)

    'CSA': (59.3 + 59.1 + 39.0 + 34.8) * 10**(-4), # double, Muscle CSA, m^2, Quad, Handsfield et a. 2014

    # Max and min values for the fibre-type composition
    # Will be muscle specific and used for the motor unit recruitment model
    'alpha_f_min': 0.0,                            # double, Minimum fraction of fast fibres, unitless, Johnson et al. 1973, quads 
    'alpha_f_max': 0.5,                            # double, Maximum fraction of fast fibres, unitless, Johnson et al. 1973, quads

    ############################################
    # Heat rate parameters based on the Barclay 1996 data
    'r1_min': 0.6177, # 1/s
    'r2_min': 0.2342, # 
    'r1_max': 2.7919, # 1/s
    'r2_max': 0.697, #

    # Heat rate parameters based on the Lichtwark and Barclay 2010 data
    # 'r1_min': 0.7355, # 1/s
    # 'r2_min': 0.0897, # 
    # 'r1_max': 5 * 0.7355, # 1/s
    # 'r2_max': 5 * 0.0897, #
    
    # Heat rate parameters based on the Barclay et al. 2010 data
    # 'r1_min': 0.2845, # 1/s
    # 'r2_min':  0.11290, # unitless
    # 'r1_max': 1.12860, # 1/s
    # 'r2_max': 0.0738, # unitless

    ############################################
    # SEE parameters
    # Can either set teh stiffness in the SEE or the maximum strain
    'k_see': 6,                               # double, Stiffness of the SEE, 1/m, 
    'max_strain': 0.147,                      # double, Max strain in the CE, unitless, Used to calculate the stiffness of the tendon

    ############################################
    # Account for motor unit recruitment
    # (not used in KLD 2024)
    'scale_method': 'None',                 # Use no recruitment
    'act_T': 0.1,                           # Threshold at which fast fibres begin to be recruited
    'act_max': 0.8,                         # Value at which all fast fibres are active 

    ############################################
    # Changes to the force-length properties
    's_l': 0,                                 # double, Shift in fasicle length, m
    'F_la_width': 0.3                        # double, Force-length width parameter, unitless
}

###############################################################
# Import some of the parameters
l_0 = params['l_0']
F_0 = params['F_0']
v_max = params['v_max']
a = params['a']

# Initialize and define frequency values
idx = 0
freq_list = np.array([0.5, 1, 1.5, 2, 2.5])
net_metabolic_rate = freq_list * 0
Energy_dict = {
        'Q_m': np.empty(shape=(len(freq_list))),
        'Q_sl': np.empty(shape=(len(freq_list))),
        'Q_tot': np.empty(shape=(len(freq_list))),
        'W_tot': np.empty(shape=(len(freq_list))),
        'E': np.empty(shape=(len(freq_list)))
    }
soldic = {}

Torque_max = 11 # Torque in Nm (van der Zee and Kuo (2021)), ONE KNEE
tend = 2        # length of simulation, s

t = np.linspace(0,tend,10000) # Define time, s

# Convert torque (Nm) and convert to normalized force from the Quad
def torque2Force(torque):

    # Force in (normalized) from the knee extensors (5cm moment arm, van der Zee and Kuo (2021))
    F_tot = torque / (0.05) / F_0 
    
    return F_tot

# Loop over the frequency conditions
for freq in freq_list:

    # Define the cyclical torque
    torque_list = Torque_max * 0.5*  (np.sin((2 * np.pi * t) *freq- np.pi/2) + 1) # Nm
    
    # Convert torque to force
    F = torque2Force(torque_list) # Returns force normalized to F_0
    
    print(f"#################################################")
    print(f"Frequency = {freq}")

    # Run the model
    soldic[str(freq)], Energy_data = runModel(t,F, params)

    # Process the energy data 
    Energy_dict['Q_m'][idx] = Energy_data['Q_m'][-1]
    Energy_dict['Q_sl'][idx] = Energy_data['Q_sl'][-1]
    Energy_dict['Q_tot'][idx] = Energy_data['Q_tot'][-1]
    Energy_dict['W_tot'][idx] = Energy_data['W_tot'][-1]
    Energy_dict['E'][idx] = Energy_data['E'][-1]
    dHdt_vals = soldic[str(freq)].dHdt_vals
    max_act = soldic[str(freq)].max_act
    mean_act = soldic[str(freq)].mean_act
    e_ce_model = soldic[str(freq)].e_ce
    net_metabolic_rate[idx] = np.mean(dHdt_vals)

    # Output results
    print(f"Net metabolic rate = {net_metabolic_rate[idx]}")
    print(f'Maximum force = {max(F)}')
    print(f"Maximum activation = {max_act}")
    print(f"Mean activation = {mean_act}")
    print(f"CE Stretch Difference = {max(e_ce_model) - min(e_ce_model)}")
    idx = idx +  1
    
    # Save the simulation mechanics
    # dict_save = {
    #     't': t, 
    #     'act': soldic[str(freq)].act_vals, 
    #     'e_ce': soldic[str(freq)].e_ce,
    #     'dedt_ce': soldic[str(freq)].dedt_ce,
    #     'F': F
    # }
    # df_save = pd.DataFrame.from_dict(dict_save)
    # df_save.to_csv(f'./Results/vanderZee_Mech_F={freq}.csv') # Specify save location, change as needed

######################################################
######################################################
# Plotting
######################################################
######################################################
# Energy rate plot 
fig_energy,ax = plt.subplots(layout='constrained')

# Import van der Zee data
df = pd.read_csv("Data/vanderZee_MetabolicCost_data.txt")
freq = np.array(df["freq"])

# Scale rates for one leg
exp_scale = 0.5 
dHdt_exp = np.array(df["dHdt"] * exp_scale)

# Plot absolute energetic rates
ax.plot(freq_list,net_metabolic_rate,color=palette[0],linestyle='solid',marker='o',label="Model")
ax.plot(freq,dHdt_exp,color=palette[0],linestyle='dotted',marker='o',label="Experiment")
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:3.1f}'))
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Energetic Rate (W)",color=palette[0],fontsize = 8)
ax2=ax.twinx()

# Plot relative energetic rates
ax2.plot(freq_list,net_metabolic_rate/max(net_metabolic_rate)\
            ,linestyle='solid',marker='o',color=palette[-1],label="Model")
ax2.plot(freq,dHdt_exp/max(dHdt_exp)\
            ,linestyle='dotted',marker='o',color=palette[-1],label="Experiment")
ax2.set_ylim(0.2,1.05)
ax2.set_ylabel("Energetic Rate (Normalized to Max)",color=palette[-1],fontsize = 8)

leg = ax2.legend()
leg.legend_handles[0].set_color('k')
leg.legend_handles[1].set_color('k')

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# fig_energy.savefig("Figures/vK2021LichtwarkEnergetics.jpg")
# fig_energy.savefig("Figures/vK2021LichtwarkEnergetics.svg", format='svg')
# plt.show()
# plt.close()

######################################################
# FV Relationship Plot
fig_fv,ax = plt.subplots(layout='constrained')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:3.2f}'))

yshift = 0.05 # Line shift factor
e_ce = np.linspace(-1.25,1.25)
lwidth = 10

def F_va(dedt_ce):
    return (1+dedt_ce/v_max)/(1-dedt_ce/v_max/a) * (dedt_ce < 0) \
            + (1.5 - 0.5*(1-dedt_ce/v_max)/(1+7.56 * dedt_ce/v_max / a))*(dedt_ce >0)\
            + (dedt_ce == 0)

ax.plot(e_ce,F_va(e_ce),color='k',label="F-V curve")

for freq in np.flip(freq_list):
    _label = "$f$ = " + str(freq)
    ax.plot(soldic[str(freq)].dedt_ce, F_va(soldic[str(freq)].dedt_ce),color = palette[int(freq * 2)-1], linewidth=lwidth, label=_label)
    lwidth += -2

ax.legend(borderpad=1)
ax.set_xlabel("CE Strain Rate")
ax.set_ylabel("Force (Normalized)")
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# fig_fv.savefig("Figures/vK2021LichtwarkEnergetics_FV_regions.jpg")
# fig_fv.savefig("Figures/vK2021LichtwarkEnergetics_FV_regions.svg", format='svg')
# plt.show()
# plt.close()

######################################################
# Mechanics figure
fig_mech,ax = plt.subplots(layout='constrained')
ax2=ax.twinx()

# Get the indexs to normalize the time domain
Nt = tend/(t[1] -t[0]) 
for freq in freq_list:
    c = markerlist[int(freq * 2-1)]
    num_cyc = tend * freq 
    idx_end = int(np.floor(Nt/num_cyc))
    t_norm = t[0:idx_end]/t[idx_end]
    
    act = soldic[str(freq)].act
    e_m = soldic[str(freq)].e_ce

    ax.plot(t_norm,act[0:idx_end],linestyle = c,color=palette[0],label=f"$f$ = {freq}")
    if freq ==0.5:
        ax2.plot(t_norm,e_m[0:idx_end]-1,linestyle = c,color=palette[-1])

ax.set_xlabel("Time (Normalized)")
ax.set_ylabel("Activation",color=palette[0])
ax2.set_ylabel("CE Strain",color=palette[-1])
ax2.set_yticks((0.0,-0.05,-0.1,-0.15))

ax.legend(loc="right")

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# fig_mech.savefig("Figures/vK2021LichtwarkMech_Cycle.jpg")
# fig_mech.savefig("Figures/vK2021LichtwarkMech_Cycle.svg", format='svg')
# plt.show()
# plt.close()

######################################################
# Stacked bar chart 
x = np.arange(len(freq_list))  # the label locations
width = 0.75 # the width of the bars
multiplier = 0

# Construct data variable
Stacked_dict = {
    "$Q_m$": Energy_dict["Q_m"],
    "$Q_{sl}$": Energy_dict["Q_sl"]
}

fig, ax = plt.subplots(layout='constrained')
bottom = np.zeros(5)

idx = 0
for attribute, measurement in Stacked_dict.items():
    rects = ax.bar(x, measurement, width, label=attribute, bottom=bottom, color=palette[idx])
    bottom += measurement
    idx +=4

ax.set_ylabel('Energy (J)')
ax.set_xlabel('Frequency (Hz)')
ax.set_xticks(x,("0.5","1.0","1.5","2.0","2.5"))

ax.legend(loc='upper left',ncol=5)

ax.yaxis.set_major_formatter(StrMethodFormatter('{x:3.1f}'))

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# plt.savefig('Figures/vK2021LichtwarkEnergetics_Stacked_Integ.jpg')
# plt.savefig("Figures/vK2021LichtwarkEnergetics_Stacked_Integ.svg", format='svg')
plt.show()

#####################################################
#####################################################
# Export data to Results folder 
#####################################################
#####################################################
# dict_save = {
#     'freq_list': freq_list,
#     'dHdt': net_metabolic_rate
# }
# df_save = pd.DataFrame.from_dict(dict_save)
# df_save.to_csv('./Results/vanderZee_dHdt_None.csv') # Specify save location, change as needed
