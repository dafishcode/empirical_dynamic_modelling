import admin_functions as adfn
import EDM as efn
#----------------------------------------------------------------------
Fcode = '/nadata/mnlsc/home/dburrows/Documents/empirical_dynamic_modelling/'
Fdata = '/nadata/mnlsc/home/dburrows/Documents/PTZ-WILDTYPE-CCM/'



#=================================
def CCM_sort(co, tr, dff, name):
#=================================

    """
    This function sorts all trace and coord data into dictionary for CCM and 2d arrays as hdf5 files

    Inputs:
        co (np array): cells x XYZ coordinates and labels
        trace (np array): cells x timepoints, raw fluorescence values
        dff (np array): cells x timepoints, normalised fluorescence values
        name (str): fish name for saving
    
    Returns:
        f_dict (dict): dictionary containing coordinates, mean trace, trace, mean dff, and dff for all neurons in brain.
        
    
    
    """
    import numpy as np
    import h5py
    
    def np2h5(full_name, array):
        #Convert numpy array to hdf5 file

        f = h5py.File(full_name, 'w')
        f.create_dataset("data", data = array)
        f.close()

    
    
    f_dict = {}
    #Check that trace and dff files are the same length
    if co.shape[0] == tr.shape[0] and tr.shape[0] == dff.shape[0]:
        sub_tr, sub_co = adfn.select_region(tr, co, 'all')
        mean_tr =  np.apply_along_axis(np.mean, 0, sub_tr)

        sub_dff, sub_co = adfn.select_region(dff, co, 'all')
        mean_dff =  np.apply_along_axis(np.mean, 0, sub_dff)
        
        
        #PUT INTO DICT
        f_dict = { 'coord': sub_co, 'mean trace' : mean_tr, 'trace': sub_tr, 'mean dff' : mean_dff, 'dff' : sub_dff}


        # CONVERT TO HDF5
        trace_list = ['trace', 'dff']
        #loop through each trace type
        for i in trace_list:
            full_name = name + '_' + i + '_pre-CCM.h5'
            #concatenate with mean trace at the top for kEDM
            array = np.vstack((f_dict['mean ' + i], f_dict[i])).T
            np2h5(full_name, array)
              
        np.save(Fdata + '/' + mode + '/' + name + '_pre-CCM.npy', f_dict)
            
        return(f_dict)

    else:
        print("data wrong shape")
        return()
    



#=====================================
def kspace_meantrace(coord, trace, k):
#=====================================
    """
    This function performs kmeans on spatial coordinates and averages cell traces. 
    
    Inputs:
        coord (np array): X x Y x Z coordinates
        trace (np array): signal x time
        k (int): k clusters
    
    Returns:
        k_coord (np array): space x dimension for clustered cells
        k_trace (np array): signal x time for clustered cells

    """

    from scipy.cluster.vq import kmeans2
    import numpy as np
    
    k_coord, k_lab = kmeans2(coord, k) #Perform k means
    k_coord = k_coord[np.unique(k_lab)] #remove empty clusters

    k_trace = np.zeros((len(np.unique(k_lab)),trace.shape[1]))  
    for x,n in enumerate(np.unique(k_lab)): #loop through all clusters
        sub_trace = trace[k_lab == n] #Find traces of clustered coords 
        k_trace[x] = np.apply_along_axis(np.mean, 0, sub_trace)
    
    return(k_coord, k_trace)
    



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


