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
    



#===============================
def simplex_project(data, E, tau, t_range):
#===============================

    """
    This function performs simplex projection over t time steps. Briefly, it splits a time series in library and prediction sets.
    It then embeds the library manifold in E dimensions, using E time lags. For each point p (embedded in E dim space) in the 
    prediction timeseries, the algorithm finds the E+1 nearest neighbours in the library manifold which forms a simplex around 
    the point p. The algorithm then iterates t time steps into the future, predicting the position of point p at t using the 
    positions of each neighbour at t exponentially weighted by the distances from p at t0. Finally the correlation between true
    and predicted values are returned. See: Sugihara et al. 'Nonlinear Forecasting as a way of distinguishing chaos from measurement
    error in time series'.
    
    
    Inputs:
        data (np array): 1d vector of time series to perform simplex projection. 
        E (int): embedding dimension
        tau (int): time delay to use for embedding
        t_range (int): how many time steps ahead to predict
        
    
    Returns:
        corr_vec (list): list of correlations for each t
        
    """

    from scipy import spatial
    import numpy as np

    corr_list = list_series(2, t_range) # for each time prediction there should be a 2d list to put in real and pred values


    # split data in half into library and prediction
    lib = data[:data.shape[0]//2]
    pred = data[data.shape[0]//2:]

    # Build manifold with given E and tau
    lib_m = lfn.takens_embed(E, tau, lib)
    pred_m = lfn.takens_embed(E, tau, pred)

    #find the E+1 nearest neighbours in library
    dist_mat = spatial.distance_matrix(pred_m, lib_m) #compute distances between all points
    nn_num = E+1 #how many nearest neighbours to find


    #Loop through each point in pred
    for num in range(pred_m.shape[0]):

        # Find nearest neighbours in library for each pred_m point
        current_point = pred_m[num]
        curr_dist = dist_mat[num]
        nn_ind = sorted(range(len(curr_dist)), key=lambda k: curr_dist[k])[:nn_num] #return indeces of nearest neighbours in library

        #Calculate weights for simplex projection - weights are calculated from nn distance at t0
        nn = lib_m[nn_ind] #positions of nearest neighbours in library, to current point in pred at t0
        nn_dist = dist_mat[num, nn_ind]  #distances of each nn to our pred point
        w_mat = np.exp(-1*(nn_dist/np.min(nn_dist))) #matrix of weights for each nn 

        # Loop in time and predict
        for t in range(1, t_range+1):

            # Where do nn end up at t + n
            nn_ind_tp = np.array(nn_ind) + t #find indeces of neighbours in the future for simplex projection

            if sum(nn_ind_tp >= lib_m.shape[0]) >0: #ignore points whose boundaries go outside the data 
                continue 

            nn_tp = lib_m[nn_ind_tp] # locations of neighbours in future

            #Simple project - how much do the positions of neighbours relative to point of interest change over time 
            #use weights from t 0
            #use neighbour points from t + n
            x_tp = pred[num] #Point I am trying to predict

            x_tp_pred = 0 
            for nn_i in range(w_mat.shape[0]): #Loop through each nn and sum over the weight*position at tp
                x_tp_pred+= (w_mat[nn_i]/np.sum(w_mat))*nn_tp[nn_i]
            x_tp_pred = x_tp_pred[0] #project back into 1d space

            corr_list[t-1][0] = np.append(corr_list[t-1][0], x_tp) #true
            corr_list[t-1][1] = np.append(corr_list[t-1][1], x_tp_pred) #estimated
        
        #Calculate correlation coefficient
        corr_vec = []
        for f in range(len(corr_list)): corr_vec = np.append(corr_vec, np.corrcoef(corr_list[f][0][1:],corr_list[f][1][1:])[0][1])
    return(np.array(corr_vec))
    
    
#===============================
def find_E_simplex(data, tau, E_range, t_range):
#===============================

    """
    This function runs simplex projection over a range of E values with a given tau, 
    and returns the E with greatest correlation between real variable and predicted. 
    
    
    Inputs:
        data (np array): 1d vector of time series to perform simplex projection. 
        tau (int): time delay to use for embedding
        E_range (int): how many embedding dimensions to check
        t_range (int): how many time steps ahead to predict
        
    
    Returns:
        E (int): optimal embedding dimension that best unfolds the manifold

        
    """
    import numpy as np

    corr_l = [0]*E_range
    for E in range(1, E_range+1):
        corr_l[E-1] = simplex_project(data, E, tau, t_range)
        print('Done E = ' + str(E))

    E = np.where(corr_l == np.max(corr_l))[0][0] + 1
    return(E)



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


