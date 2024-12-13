'''
Script to compute energy rates based on force traces from Beck et al. (2022).

Author: Ryan Konno, University of Queensland
        r.konno@uq.edu.au
'''
###############################################################
# Import
import numpy as np
import pandas as pd

# Import matplotlib
import matplotlib.pyplot as plt
font = {'size'   : 8}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = [9.0/2.56, 8.0/2.56]
plt.rcParams['figure.dpi'] = 150
plt.rcParams["svg.fonttype"] = "none"

# Import the combined mech heat model 
from MuscleModel.MuscleModel import runModel

###############################################################
# Define plotting data (color and marker list)
palette = ['#066b0d',  '#838907', '#ffa600']
markerlist = ('solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5, 1, 5)))

###############################################################
params = {
    'Exp': 'Beck2022',

    'l_0': 0.041,                   # double, Muscle fasicle length, m, (Beck et al. 2022)
    'F_0': 3100,                    # double, Maximum isometric force, N, SOL, (Handsfield)

    'v_max': 4.4,                   # double, Maximum shortening rate, 1/s, Bohm et al. 2019
    'a': 0.17,                      # double, Force-velocity relationship curvature, unitless, Dick et al. 2017
    'mass': 275,                    # double, Muscle mass, g, (Ward et al. 2009, Soleus)

    # Max and min values for the fibre-type composition
    # Will be muscle specific and used for the motor unit recruitment model
    'alpha_f_min': 0.0,             # double, Minimum fraction of fast fibres, unitless, Johnson et al. 1973
    'alpha_f_max': 0.15,            # double, Maximum fraction of fast fibres, unitless, Johnson et al. 1973
        
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
    'k_see': 50,                     # double, SEE stiffness, unitless, Chosen to obtain correct e_ce # Works for F_0 = 3100

    ############################################
    # Account for motor unit recruitment
    # (not used in KLD 2024)
    'scale_method': 'None',          # String, Specify the scaling of the energy rates 
    'act_T': 0.0,                    # double, Threshold at which fast fibres begin to be recruited
    'act_max': 1.0,                  # double, Value at which all fast fibres are active
    
    ############################################
    # Changes to the force-length properties
    's_l': 0,                        # double, Shift in fasicle length, m
    'F_la_width': 0.15,              # double, Force-length width parameter, unitless

    ############################################
    # Recovery heat
    'include_recovery': True,                  # Include recovery heat
    'recovery_ratio': 1,                     # Ratio of initial enthalpy to recovery heat
}

###############################################################
# Define parameters
l_0 = params['l_0']
F_0 = params['F_0']
v_max = params['v_max']
a = params['a']

# Force output (based on Beck et al. 2020)
duty = 1
F_amp = 300/F_0 # Base on max force of (300 N)

# Define the time 
tend = 1.2
t = np.linspace(0,tend,1000)

# Generate force trace to mimic Beck et al. 2022
t1 = 0.4
t2 = 0.8
sineramp = F_amp * 0.5 * (np.sin(np.pi*t / (t1) - np.pi/2) + 1)
negsineramp = F_amp * 0.5 * (np.sin(np.pi*(t-t1) / (t2 - t1) + 3*np.pi/2) + 1)
F = sineramp * (t < t1) + F_amp * (t >= t1) * (t <= t2) + negsineramp * (t > t2)

dt = t[1] - t[0]

# List of stretch factors, scaled to units of m
s_l_list = np.array((0, 0.1, 0.2)) * l_0

# Initialize variables
soldic = {}
net_metabolic_rate = np.array(s_l_list) * 0
Energy_dict = {
        'Q_m': np.empty(shape=(len(s_l_list))),
        'Q_sl': np.empty(shape=(len(s_l_list))),
        'Q_tot': np.empty(shape=(len(s_l_list))),
        'W_tot': np.empty(shape=(len(s_l_list))),
        'E': np.empty(shape=(len(s_l_list)))
    }

print('##############################################')

# Loop over shifts in fascicle length
for i in range(len(s_l_list)):
    s_l = s_l_list[i]
    exp_type = str(s_l/l_0)

    params['s_l'] = s_l

    # Run model
    soldic[exp_type], Energy_data = runModel(t, F, params)

    dHdt_vals = soldic[exp_type].dHdt_vals
    max_act = soldic[exp_type].max_act
    mean_act = soldic[exp_type].mean_act
    net_metabolic_rate[i] = np.mean(dHdt_vals)

    # Process the energy data 
    Energy_dict['Q_m'][i] = Energy_data['Q_m'][-1]
    Energy_dict['Q_sl'][i] = Energy_data['Q_sl'][-1]
    Energy_dict['Q_tot'][i] = Energy_data['Q_tot'][-1]
    Energy_dict['W_tot'][i] = Energy_data['W_tot'][-1]
    Energy_dict['E'][i] = Energy_data['E'][-1]

    print(f's_f = {s_l/l_0}')
    print(f'Metabolic rate: {np.mean(soldic[exp_type].dHdt_vals)}W')
    print(f'Max Activation: {soldic[exp_type].max_act}')
    print(f'Max length: {max(soldic[exp_type].e_ce) * l_0}')
    print(f'Min length: {min(soldic[exp_type].e_ce) * l_0}')
    print(f'Peak strain: {1-min(soldic[exp_type].e_ce)}')
    print(f'Peak strain rate: {min(soldic[exp_type].dedt_ce)}')
    print('______________________________________________')

    # Save the simulation mechanics
    # dict_save = {
    #     't': t, 
    #     'act': soldic[exp_type].act_vals, 
    #     'e_ce': soldic[exp_type].e_ce,
    #     'dedt_ce': soldic[exp_type].dedt_ce,
    #     'F': F
    # }
    # df_save = pd.DataFrame.from_dict(dict_save)
    # df_save.to_csv(f'./Results/Beck2022_Mech_{exp_type}.csv') # Specify save location, change as needed

#####################################################
#####################################################
# Plot the results
#####################################################
#####################################################
# Energy plot
figure, ax = plt.subplots(layout='constrained')
ax2 = ax.twinx()

# Import data
data = pd.read_csv("Data/Beck2022_MetabolicCost_Data.txt")

# Scale data
exp_scale = 0.6
dHdt_exp_scaled = exp_scale * np.array(data["dHdt"])
dHdt_model = np.array((np.mean(soldic[str(s_l_list[0]/l_0)].dHdt_vals),np.mean(soldic[str(s_l_list[1]/l_0)].dHdt_vals),np.mean(soldic[str(s_l_list[2]/l_0)].dHdt_vals)))

# Plot data
ax.plot(s_l_list,dHdt_model\
            ,linestyle='solid',marker='o',color=palette[0],label="Model")
ax.plot(s_l_list,dHdt_exp_scaled\
            ,linestyle='dotted',marker='o',color=palette[0],label="Experiment")
ax.set_xlabel("$s_f$ (m)")
ax.set_ylabel("Energetic Rate (W)", color = palette[0])

# Plot scaled values
ax2.plot(s_l_list,dHdt_model/max(dHdt_model)\
            ,linestyle='solid',marker='o',color=palette[-1],label="Model scaled")
ax2.plot(s_l_list,dHdt_exp_scaled/max(dHdt_exp_scaled)\
            ,linestyle='dotted',marker='o',color=palette[-1],label="Experiment scaled")
ax2.set_ylabel("Energetic Rate (Normalized to Max)",color=palette[-1],fontsize = 8)

leg = ax.legend()
leg.legend_handles[0].set_color('k')
leg.legend_handles[1].set_color('k')

# figure.savefig("Figures/Beck2022LichtwarkEnergetics.jpg")
# figure.savefig("Figures/Beck2022LichtwarkEnergetics.svg", format='svg')
# plt.show()
# plt.close()

########################################################
# FL regions 
fig,ax = plt.subplots(layout='constrained')

yshift = 0.0 # Line shift factor
e_ce = np.linspace(0.5,1.1)
def F_la(e_m):
    skew = 0.6
    width = params["F_la_width"] # orginal was 0.3 (altered for this experiment to get realistic values)
    round = 2.3
    return np.exp(-np.abs((e_m**skew -1)/width)**round)
ax.plot(e_ce,F_la(e_ce),color='k',label="F-L curve")
idx = 0
for s_l in s_l_list:
    _label = "$s_f$ = " + str(s_l/l_0)
    ax.plot(soldic[str(s_l/l_0)].e_ce, soldic[str(s_l/l_0)].F_la + yshift,color=palette[idx], linewidth = 6, label=_label)
    idx += 1
ax.legend()
ax.set_xlabel("CE Strain")
ax.set_ylabel("Force (Normalized)")

# plt.savefig("Figures/Beck2022_FL_regions.jpg")
# plt.savefig("Figures/Beck2022_FL_regions.svg", format='svg')
# plt.close()

########################################################
# Activation plot
fig,ax = plt.subplots(layout='constrained')
ax2 = ax.twinx()
markerlist = ('solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5, 1, 5)))

idx = 0

for s_l in s_l_list:
    _label = "$s_f$ = " + str(s_l)
    ax.plot(t, soldic[str(s_l/l_0)].act_vals,linestyle = markerlist[idx],color=palette[0], label=_label)
    ax2.plot(t, soldic[str(s_l/l_0)].e_ce,linestyle = markerlist[idx],color=palette[-1], label=_label)
    idx +=1

ax.set_xlabel("Time (s)")
ax.set_ylabel("Activation (Normalized)",color=palette[0])
ax2.set_ylabel("CE Strain",color=palette[-1])

leg = plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
leg.legend_handles[0].set_color('k')
leg.legend_handles[1].set_color('k')
leg.legend_handles[2].set_color('k')

# plt.savefig("Figures/Beck2022_MechPlot.jpg")
# plt.savefig("Figures/Beck2022_MechPlot.svg", format='svg')

########################################################
# Stacked bar chart
x = np.arange(len(s_l_list))  # the label locations
width = 0.5 # the width of the bars
multiplier = 0

# Contruct data 
Stacked_dict = {
    "$Q_m$": Energy_dict["Q_m"],
    "$Q_{sl}$": Energy_dict["Q_sl"]
}

fig, ax = plt.subplots(layout='constrained')
bottom = np.zeros(3)

idx = 0

for attribute, measurement in Stacked_dict.items():
    rects = ax.bar(x,  measurement, width, label=attribute, color=palette[idx],bottom=bottom)
    bottom +=  measurement
    idx = idx + 2 # Get last color in palette

ax.set_ylabel('Energy (J)')
ax.set_xlabel('$s_f$ (m)')
ax.set_xticks(x,s_l_list)
ax.legend(loc='upper left',ncol=3)

# plt.savefig('Figures/Beck2022_Stacked_Energy.jpg')
# plt.savefig("Figures/Beck2022_Stacked_Energy.svg", format='svg')
plt.show()
# plt.close()

#####################################################
#####################################################
# Export data to Results folder 
#####################################################
#####################################################
# Organize solution dictionary with results to save 
# dict_save = {
#     's_l_list': s_l_list,
#     'dHdt': net_metabolic_rate
# }
# df_save = pd.DataFrame.from_dict(dict_save)
# df_save.to_csv('./Results/B2022_dHdt_.csv') # Specify save location, change as needed