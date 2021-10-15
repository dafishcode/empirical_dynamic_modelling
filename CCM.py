import admin_functions as adfn

#=================================
def CCM_sort(co, tr, dff):
#=================================

    """
    This function sorts all trace and coord data into dictionary for CCM. 

    Inputs:
        co (np array): cells x XYZ coordinates and labels
        trace (np array): cells x timepoints, raw fluorescence values
        dff (np array): cells x timepoints, normalised fluorescence values
        name (str): save name 
    
    Returns:
        f_dict (dict): dictionary containing coordinates, mean trace, trace, mean dff, and dff for all neurons in brain. 
    
    """
    import numpy as np
    
    
    f_dict = {}

    if co.shape[0] == tr.shape[0] and tr.shape[0] == dff.shape[0]:
        sub_tr, sub_co = adfn.select_region(tr, co, 'all')
        mean_tr =  np.apply_along_axis(np.mean, 0, sub_tr)

        sub_dff, sub_co = adfn.select_region(dff, co, 'all')
        mean_dff =  np.apply_along_axis(np.mean, 0, sub_dff)

        f_dict = { 'coord': sub_co, 'mean trace' : mean_tr, 'trace': sub_tr, 'mean dff' : mean_dff, 'dff' : sub_dff}
        return(f_dict)

    else:
        print("data wrong shape")
        return()










#===========================
def E_ccm_heatmap(E, ccm, n_bins):
#===========================
    """
    Heatmap of embedding vs ccm rho for each neuron, calculated by estimating a histogram of values along the CCM axis and flatteing it as a vector. 
    
    Inputs:
    E (np array): 1d vector of embedding dimension for each neuron
    ccm (np array): cells x cells, pairwise CCM prediction for each neuron 
    n_bins (int): number of bins for histogram
    
    
    Returns:
    hist (np array): nbins x nbins with absolute counts in each cell
    
    """
    import numpy as np
    
    unq = np.unique(E) 
    hist = np.zeros((len(unq), n_bins))
    count=0
    for i in unq:
        hist[count] = np.histogram(np.ravel(ccm[np.where(E == i)]), bins = np.linspace(0, 1, n_bins+1))[0]
        count+=1
    return(hist)