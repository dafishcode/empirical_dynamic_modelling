import admin_functions as adfn

Fdata = '/Users/dominicburrows/Dropbox/PhD/analysis/Project/PTZ-WILDTYPE-CCM/'

#=================================
def CCM_sort(co, tr, dff, name, mode):
#=================================

    """
    This function sorts all trace and coord data into dictionary for CCM and 2d arrays as hdf5 files

    Inputs:
        co (np array): cells x XYZ coordinates and labels
        trace (np array): cells x timepoints, raw fluorescence values
        dff (np array): cells x timepoints, normalised fluorescence values
        name (str): fish name for saving
        mode (str): 'single' assumes file is continuous and saves as a single file; 'comb_3' assumes file should be split in 3. 
    
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
        
        
        if mode == 'single':
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
                
                
        if mode == 'comb_2':
            num = 9009
                
                #PUT INTO DICT
                #THIS IS MADE FOR THE FIRST 5MINS BEING CUTOFF - CHANGE IF NEEDED!!!!
            f_dict = { 'coord': sub_co, 'BLN mean trace' : mean_tr[:num], 'BLN trace': sub_tr[:,:num], 'BLN mean dff' : mean_dff[:num], 'BLN dff' : sub_dff[:,:num],
                         'PTZ05 mean trace' : mean_tr[num:], 'PTZ05 trace': sub_tr[:,num:], 'PTZ05 mean dff' : mean_dff[num:], 'PTZ05 dff' : sub_dff[:,num:]}
            
            
            # CONVERT TO HDF5
            trace_list = ['trace', 'dff']
            cond_list = ['BLN', 'PTZ05']
            #loop through each trace type
            for i in trace_list:
                for e in cond_list:
                    full_name = name[0:-6] + e + "_" + name[-6:] + '_' + i + '_pre-CCM.h5'
                    #concatenate with mean trace at the top for kEDM
                    array = np.vstack((f_dict[e + ' mean ' + i], f_dict[e + ' ' + i])).T
                    np2h5(full_name, array)
    
           
        
        if mode == 'comb_3':
            #Check that each segment has the same length
            if sub_tr.shape[1]%9828 == 0 and sub_dff.shape[1]%9828 == 0:
                
                num = 9828
                
                #PUT INTO DICT
                f_dict = { 'coord': sub_co, 'BLN mean trace' : mean_tr[:num], 'BLN trace': sub_tr[:,:num], 'BLN mean dff' : mean_dff[:num], 'BLN dff' : sub_dff[:,:num],
                         'PTZ05 mean trace' : mean_tr[num:2*num], 'PTZ05 trace': sub_tr[:,num:2*num], 'PTZ05 mean dff' : mean_dff[num:2*num], 'PTZ05 dff' : sub_dff[:,num:2*num],
                         
                         'PTZ20 mean trace' : mean_tr[2*num:], 'PTZ20 trace': sub_tr[:,2*num:], 'PTZ20 mean dff' : mean_dff[2*num:], 'PTZ20 dff' : sub_dff[:,2*num:]}
            
            
                # CONVERT TO HDF5
                trace_list = ['trace', 'dff']
                cond_list = ['BLN', 'PTZ05', 'PTZ20']
                #loop through each trace type
                for i in trace_list:
                    for e in cond_list:
                        full_name = name[0:-6] + e + "_" + name[-6:] + '_' + i + '_pre-CCM.h5'
                        #concatenate with mean trace at the top for kEDM
                        array = np.vstack((f_dict[e + ' mean ' + i], f_dict[e + ' ' + i])).T
                        np2h5(full_name, array)

            
            else:
                print('Data has non equal condition lengths')
                return()
              
                
             
        np.save(Fdata + '/' + mode + '/' + name + '_pre-CCM.npy', f_dict)
            
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