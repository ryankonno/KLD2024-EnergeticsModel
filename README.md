# KLD2024-EnergeticsModel
Mathematical model for investigating the ability of physiologically-based phenomenological models to predict in vivo muscle energetics.

This repository contains the code for the manuscript 

Konno RN, Lichtwark GA, and Dick TJM. Using physiologically-based models to predict \textit{in vivo} skeletal muscle energetics. 2024. In preparation.

There are three main code files used to run the model: vanderZee2021_LichtwarkModel.py, Beck2020_LichtwarkModel.py, and Beck2022_LichtwarkModel.py. These codes are designed to predict energy use based on the experimental data from the following three papers respectively 
 
 - van der Zee, T. J., & Kuo, A. D. (2021). The high energetic cost of rapid force development in muscle. Journal of Experimental Biology, 224(9). https://doi.org/10.1242/JEB.233965/237823

 - Beck, O. N., Gosyne, J., Franz, J. R., & Sawicki, G. S. (2020). Cyclically producing the same average muscle-tendon force with a smaller duty increases metabolic rate. Proceedings of the Royal Society B: Biological Sciences, 287(1933). https://doi.org/10.1098/RSPB.2020.0431

 - Beck, O. N., Trejo, L. H., Schroeder, J. N., Franz, J. R., & Sawicki, G. S. (2022). Shorter muscle fascicle operating lengths increase the metabolic cost of cyclic force production. Journal of Applied Physiology, 133(3), 524–533. https://doi.org/10.1152/JAPPLPHYSIOL.00720.2021

The MuscleModel directory contains the muscle model including the mechanical (MechModel.py) and energetic (HeatModel.py) components of the model. Each of these codes are called by MuscleModel.py.

**NOTE:** There is an implementation of a simplified motor unit recruitment model (RecruitmentModel.py) included in MuscleModel. By default it is not used, and it is **not** used in the KLD 2024 manuscript. To utilize this model switch the parameter 'scale_method' from 'None' to 'fibre-act' (scales the energetic parameters based on the muscle activations). 