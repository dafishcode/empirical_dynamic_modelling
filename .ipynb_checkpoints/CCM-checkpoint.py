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