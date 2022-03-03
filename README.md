# empirical_dynamic_modelling
Repo for running empirical dynamic modelling techniques on data - using lagged coordinate embedding to perform simplex projection and convergent cross mapping (CCM) - techniques which can account for non-linear dynamics to reconstruct attractors from time series data. 

See Sugihara et al. 
CCM - https://www.science.org/doi/abs/10.1126/science.1227079

Simplex projection - https://www.nature.com/articles/344734a0

## What is this repo for?
* the implementation of lagged coordinate embedding on time series data
* the implementation of simplex projection
* the implementation of convergent cross mapping (CCM) on time series data
* the analysis and visualisation of CCM results 

## What does this repo contain?
* Modules contain functions for running convergent cross mapping, lagged coordinate embedding and evaluating CCM results
* Accompanying ipynotebooks demonstrate how to use the modules

### Modules
'admin_functions.py' - useful administrative functions 

'LCE.py' - functions for performing lagged coordinate embedding

'CCM.py' - functions for implementing convergent cross mapping

'kedm_script.sh' - shell script for batch running kedm on salk system

### Notebooks

'LCE.ipynb' - running and implementing LCE algorithm

'CCM_run.ipynb' - running and implementing CCM algorithm

'CCM_eval.ipynb' - evaluating and analysing CCM results






