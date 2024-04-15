'''
Script to compute energy rates based on force traces from Beck et al. (2020).

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
    'Exp': 'Beck2020',

    'l_0': 0.0386,                  # double, Muscle fasicle length, m, (Beck et al. 2020)
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

}

####################################################################
# Define parameters
F_0 = params['F_0']
v_max = params['v_max']
a = params['a']

# Define the time 
tend = 1.3
t = np.linspace(0,tend,10000) # 1s experiment (simple force calculation)

# Get duty factors from the experiment
data = pd.read_csv("Data/Beck2020_HighTorque_MetabolicCost_data.txt")
df_exp = np.array(data["df"])
df_list = df_exp 

# Compute forces
f_high_amp = 250/F_0 # From paper, High duty force was 250 N 

# Implementation based on experimental values (approximated)
f_mid_amp = 375/F_0
f_low_amp = 550/F_0

f_high_test = f_high_amp * 0.5 * (np.sin(2*np.pi*t / (tend * df_list[2]) - np.pi/2) + 1) * (t < tend *  df_list[2])
f_mid_test  = f_mid_amp  * 0.5 * (np.sin(2*np.pi*t / (tend * df_list[1]) - np.pi/2) + 1) * (t < tend *  df_list[1])
f_low_test  = f_low_amp  * 0.5 * (np.sin(2*np.pi*(t) / (tend * df_list[0]) - np.pi/2) + 1) * (t < tend * df_list[0])

# Assemble list of force values
F_list = (f_low_test,f_mid_test,f_high_test)

soldic = {}
net_metabolic_rate = np.array(df_list) * 0
print('##############################################')

Energy_dict = {
        'Q_m': np.empty(shape=(len(df_list))),
        'Q_sl': np.empty(shape=(len(df_list))),
        'Q_tot': np.empty(shape=(len(df_list))),
        'W_tot': np.empty(shape=(len(df_list))),
        'E': np.empty(shape=(len(df_list)))
    }

# Loop over the duty factors
for i in (0,1,2):
    df = df_list[i]
    params['df'] = df
    f = F_list[i]

    # Run model 
    soldic[str(df)], Energy_data = runModel(t,f, params)

    dHdt_vals = soldic[str(df)].dHdt_vals
    max_act = soldic[str(df)].max_act
    mean_act = soldic[str(df)].mean_act
    net_metabolic_rate[i] = np.mean(dHdt_vals)

    # Process the energy data 
    Energy_dict['Q_m'][i] = Energy_data['Q_m'][-1]
    Energy_dict['Q_sl'][i] = Energy_data['Q_sl'][-1]
    Energy_dict['Q_tot'][i] = Energy_data['Q_tot'][-1]
    Energy_dict['W_tot'][i] = Energy_data['W_tot'][-1]
    Energy_dict['E'][i] = Energy_data['E'][-1]

    print(f'High duty ({str(df)}):')
    print(f'Max force: {max(F_list[i]) * F_0} N')
    print(f'Metabolic rate: {np.mean(soldic[str(df)].dHdt_vals)} W')
    print(f'Max Activation: {soldic[str(df)].max_act}')
    print(f'Max fascicle strain: { 1 - min(soldic[str(df)].e_ce)}')
    print(f'Max strain rate: {min(soldic[str(df)].dedt_ce)}')
    print('______________________________________________')
    
    # Save the simulation mechanics
    # dict_save = {
    #     't': t, 
    #     'act': soldic[str(df)].act_vals, 
    #     'e_ce': soldic[str(df)].e_ce,
    #     'dedt_ce': soldic[str(df)].dedt_ce,
    #     'F': f
    # }
    # df_save = pd.DataFrame.from_dict(dict_save)
    # df_save.to_csv(f'./Results/Beck2020_Mech_duty_{df}.csv') # Specify save location, change as needed


########################################################
# Energy plot
figure, ax = plt.subplots(layout='constrained')
ax2 = ax.twinx()

# Scale data
exp_scale = 0.6
dHdt_exp = exp_scale * np.array(data["dHdt"])
dHdt_model = np.array((np.mean(soldic[str(df_list[0])].dHdt_vals),np.mean(soldic[str(df_list[1])].dHdt_vals),np.mean(soldic[str(df_list[2])].dHdt_vals)))

# Plot data
ax.plot(df_list,dHdt_model\
            ,linestyle='solid',marker='o',color=palette[0],label="Model")
ax.plot(df_list,dHdt_exp\
            ,linestyle='dotted',marker='o',color=palette[0],label="Experiment")
ax.set_xlabel("Duty Factor")
ax.set_ylabel("Energetic Rate (W)", color = palette[0])

# Plot scaled values
ax2.plot(df_list,dHdt_model/max(dHdt_model)\
            ,linestyle='solid',marker='o',color=palette[-1],label="Model scaled")
ax2.plot(df_list,dHdt_exp/max(dHdt_exp)\
            ,linestyle='dotted',marker='o',color=palette[-1],label="Experiment scaled")
ax2.set_ylabel("Energetic Rate (Normalized to Max)",color=palette[-1],fontsize = 8)

leg = ax.legend(loc='upper right', bbox_to_anchor=(0.99, 1))
leg.legend_handles[0].set_color('k')
leg.legend_handles[1].set_color('k')

# figure.savefig("Figures/Beck2020LichtwarkEnergetics.jpg")
# figure.savefig("Figures/Beck2020LichtwarkEnergetics.svg", format='svg')

# plt.show()
# plt.close()


########################################################
# Force plot
fig,ax = plt.subplots(layout='constrained')
ax2 = ax.twinx()

markerlist = ('solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5, 1, 5)))

for i in range(len(df_list)):
    ax.plot(t, soldic[str(df_list[i])].act_vals,color=palette[0],linestyle = markerlist[i],label=f'Duty: {round(df_list[i],2):<04}')
    ax2.plot(t, F_list[i],color=palette[-1],linestyle = markerlist[i]) #,label=f'Duty: {df_list[i]}')

ax.set_xlabel("Time (s)")
ax.set_ylabel("Activation",color=palette[0])
ax2.set_ylabel("Force (Normalized)",color=palette[-1])
ax2.set_yticks((0, 0.05, 0.1, 0.15))

leg = fig.legend(loc='upper right', bbox_to_anchor=(0.87, 1))
leg.legend_handles[0].set_color('k')
leg.legend_handles[1].set_color('k')
leg.legend_handles[2].set_color('k')

# fig.savefig("Figures/Beck2020_MechPlot.jpg")
# fig.savefig("Figures/Beck2020_MechPlot.svg", format='svg')

# plt.show()
# plt.close()

########################################################
# FV regions 
fig_fv,ax = plt.subplots(layout='constrained')

yshift = 0.05 # Line shift factor
e_ce = np.linspace(-0.5,0.5)
def F_va(dedt_ce):
    return (1+dedt_ce/v_max)/(1-dedt_ce/v_max/a) * (dedt_ce < 0) \
            + (1.5 - 0.5*(1-dedt_ce/v_max)/(1+7.56 * dedt_ce/v_max / a))*(dedt_ce >0)\
            + (dedt_ce == 0)
ax.plot(e_ce,F_va(e_ce),color='k',label="F-V curve")
lwidth = 10
idx = 0
for df in df_list:
    _label = "Duty: " + f'{round(df,2):<04}'
    ax.plot(soldic[str(df)].dedt_ce, soldic[str(df)].F_va,color=palette[idx], label=_label,linewidth=lwidth,)
    lwidth += -2
    idx = idx + 1

ax.legend(borderpad=1)
ax.set_xlabel("CE Strain Rate")
ax.set_ylabel("Force (Normalized)")

# fig_fv.savefig("Figures/Beck2020_FV_regions.jpg")
# fig_fv.savefig("Figures/Beck2020_FV_regions.svg", format='svg')

# plt.show()
# plt.close()


########################################################
# Stacked bar chart 
x = np.arange(len(df_list))  # the label locations
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
    rects = ax.bar(x, measurement, width,color=palette[idx], label=attribute, bottom=bottom)
    bottom += measurement
    idx = idx + 2 # Get 1st and last colors in plot

ax.set_ylabel('Energy (J)')
ax.set_xlabel('Duty factor')

df_list_round = df_list
for i in range(len(df_list)):
    df_list_round[i] = round(df_list[i],2)
ax.set_xticks(x,df_list_round)

ax.legend(loc='upper right',ncol=3)

# plt.savefig('Figures/Beck2020_Stacked_Energy.jpg')
# plt.savefig("Figures/Beck2020_Stacked_Energy.svg", format='svg')

plt.show()
# plt.close()

#####################################################
#####################################################
# Export data to Results folder 
#####################################################
#####################################################
# Organize solution dictionary with results to save 
# dict_save = {
#     'df_list': df_list,
#     'dHdt': net_metabolic_rate
# }
# df_save = pd.DataFrame.from_dict(dict_save)
# df_save.to_csv('./Results/B2020_dHdt_.csv') # Specify save location, change as needed