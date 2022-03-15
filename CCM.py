import admin_functions as adfn
import EDM as efn
#----------------------------------------------------------------------
Fcode = '/nadata/mnlsc/home/dburrows/Documents/empirical_dynamic_modelling/'
Fdata = '/nadata/mnlsc/home/dburrows/Documents/PTZ-WILDTYPE-CCM/'


#==============================================
#PREPROCESS
#==============================================
#=================================
def CCM_seizure_sort(co, tr, dff, name):
#=================================

    """
    This function sorts all trace and coord data into dictionary for CCM and 2d arrays as hdf5 files, while also adding in a meantrace to the top of the array. 
    NB - kEDM wants data structured as: time x cells
    CCM_sort function adds in a meantrace and saves traces as: cells x time in .npy dict, but time x cells in .h5 

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
    
    
    
#=================================
def CCM_trace_save(data, name):
#=================================

    """
    This function sorts all trace and coord data into dictionary for CCM and 2d arrays as hdf5 files, while also adding in a meantrace to the top of the array. 
    NB - kEDM wants data structured as: time x cells
    CCM_trace_save function renames and saves trace data as: time x cells, and 
    then saves the trace in the correct orientation in h5 format
    
    Inputs:
        data (str): file name of dataset - should be cells x timepoints, raw fluorescence values
        name (str): fish name for saving - should include datatype after run        
    
    
    """
    import numpy as np
    import h5py

    #rename for kEDM processing
    os.rename(data, name + '_pre-CCM.npy')

    def np2h5(full_name, array):
        #Convert numpy array to hdf5 file
        f = h5py.File(full_name, 'w')
        f.create_dataset("data", data = array)
        f.close()

    #save as .h5
    array = np.load(name + '_pre-CCM.npy').T
    full_name = name + '_pre-CCM.h5' 
    np2h5(full_name, array)


    
#=================================
def ccm_stats(file, mode):
#=================================

    """
    This function takes as input a filename and returns a vector of ccm statistics corresponding to each neuron.

    Inputs:
        file (str): filename - should be a '-CCMxmap.h5' file
        mode (str): what data type you want:
                        'c_to_sz' = cells that drive the seizure
                        'sz_to_c' = cells that are driven by the seizure
                        'e' = embedding dimension of each cell
                        'rd_cause' = non-linear causing neurons - mean rhodiff of neurons that are caused by neuron of interest
                        'rd_int' =  #non-linear integration - mean rhodiff of neurons that cause neuron of interest
                        
    Returns:
            (np array): 1d vector of interest (length = n cells) containing ccm stats
    
    
    """      
    
    import h5py
    import numpy as np
    data = h5py.File(file)
    
    if mode != 'c_to_sz' and mode != 'sz_to_c' and mode != 'e' and mode != 'rd_cause' and mode != 'rd_int':
        print('Mode name does not match options')
        return()
    
    if mode == 'c_to_sz':
        ccm = np.array(data['ccm']) 
        c_to_sz = ccm[1:,0] #cells that drive the seizure
        return(c_to_sz)
        
    if mode == 'sz_to_c':
        ccm = np.array(data['ccm']) 
        sz_to_c = ccm[0,1:] #cells that are driven by the seizure
        return(sz_to_c)
        
    if mode == 'e':
        e = np.array(data['e']) [1:] #embedding dimension for each neuron
        return(e)
        
    else:
        rd_m = data['rhodiff'][1:,1:] #remove seizure values
        np.fill_diagonal(rd_m,0) #remove self-ccm values 
        
        if mode == 'rd_cause':
            rd_cause = np.apply_along_axis(np.mean,1,rd_m) #non-linear causing - mean rhodiff of neurons that are caused by neuron of interest
            return(rd_cause)
            
        if mode == 'rd_int':
            rd_int = np.apply_along_axis(np.mean,0,rd_m) #non-linear integration - mean rhodiff of neurons that cause neuron of interest
            return(rd_int)
        

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




#==============================================
#RUN CCM
#==============================================

#================================    
class CCM: 
#================================    
    """
    This class runs convergent cross mapping on time series data and estimates parameters using techniques 
    from empirical dynamic modelling - e.g. simplex projection and lagged coordinate embedding. 
    
    Inputs:
        data_cause (np array): Data to perform CCM on - this should be the causative variable. 
        data_effect (np array): Data to perform CCM on - this should be the effected variable. NB can leave empty if only doing analyses for one dataset. 
    
    """
    
    #========================
    def __init__(self, data_cause, data_effect):
    #========================
        self.data_cause = data_cause 
        self.data_effect = data_effect

    #========================
    def est_tau(self, data, mode):
    #========================
        """
        This function estimates tau for lagged coordinate embedding, using different approaches. mi = find the tau that provides the first minima of the MI - this provides most independent information to initial time series without completely losing the time series. ac = find the tau at which the autocorrelation drops below 1/e. 

        Inputs:
            data (np array): 1d vector timeseries
            mode (str): 'mi' or 'ac'

        Returns:
            tau (int): estimated tau for embedding

        """
        self.tau = efn.find_tau(data, mode) #Estimate tau
        
        return(self)
        
        
    #========================
    def est_E(self, data, tau_mode, E_mode, tau_default):
    #========================
        """
        This function estimates the embedding dimension E for lagged coordinate embedding, using different approaches. 
        fnn = find the E that approaches 0 false nearest neighbours - what embedding unfolds the manifold so that nearest neighbours become preserved.
        simplex = runs simplex projection over a range of E values with a given tau, and returns the E with greatest correlation between the real variable and predicted. 

        Inputs:
            data (np array): 1d vector timeseries
            tau_mode (str): 'mi' or 'acc'
            E_mode (str): 'fnn' or 'simplex'
            'tau_default': 'yes' - means tau is set to one

        Returns:
            E (int): estimated number of dimensions to use for embedding

        """

        self.tau = self.est_tau(data, tau_mode).tau #Estimate tau
        self.E = efn.find_E(data, self.tau, E_mode) #Estimate E
    
        #in my fish calcium imaging data tau is set to 1
        if tau_default == 'yes':
            self.tau = 1 
            
        return(self)
        
    #==================================================
    def simplex(self, data, tau_mode, E_mode, tau_default):
    #==================================================
    
        """
        This function performs simplex projection over t time steps into the future.

        Inputs:
            data (np array): 1d vector timeseries
            tau_mode (str): 'mi' or 'acc'
            E_mode (str): 'fnn' or 'simplex'
            'tau_default': 'yes' - means tau is set to one


        Returns:
            simp_corr (float): correlation coefficient between observed and predicted
            simp_pred (np array): a 2d vector of observations and predictions

        """
    
        self.tau = self.est_tau(data, tau_mode).tau #Estimate tau
        self.E = self.est_E(data, tau_mode, E_mode, tau_default).E #Estimate E
        self.simp_corr, self.simp_pred = efn.simplex_project(data, self.E, self.tau, 1) #Perform simplex projection
    
        return(self)
        
    #=====================================================
    def group_embed(self, tau_mode, E_mode, tau_default):
    #=====================================================
        """
        This function performs embedding on our 2 cause and effect variables. 

        Inputs:
            tau_mode (str): 'mi' or 'acc'
            E_mode (str): 'fnn' or 'simplex'
            'tau_default': 'yes' - means tau is set to one


        Returns:
            lib_m (np array): t x E embedded time series, used to make the prediction.
            pred_m (np array): t x E embedded time series, used as the observed dataset to compare with prediction. 
        

        """
        #Initialise cause and effect dictionaries
        self.cause = {'data': self.data_cause, 'E': self.est_E(self.data_cause, tau_mode, E_mode, tau_default).E, 'tau': self.est_tau(self.data_cause, tau_mode).tau } #variable being tested as causative factor - ie pred manifold
        self.effect = {'data': self.data_effect, 'E': self.est_E(self.data_effect, tau_mode, E_mode, tau_default).E , 'tau': self.est_tau(self.data_effect, tau_mode).tau} #variable being tested as the effected variables - ie lib manifold
        
        #Embed each timeseries with respective parameters
        lib, lib_E, lib_tau = self.effect['data'], self.effect['E'], self.effect['tau'] # This is the variable that will be used to predict -  the effected variable.
        pred, pred_E, pred_tau = self.cause['data'], self.cause['E'], self.cause['tau'] # This is the variable that will be predicted - the causative variable. 
        self.embed_m = efn.takens_embed(lib_E, lib_tau, lib),efn.takens_embed(pred_E, pred_tau, pred) #lib then pred
    
        return(self)
    
    #=================================================
    def cross_map(self, tau_mode, E_mode, tau_default):
    #=================================================
        """
        This function performs cross map predictions from one manifold to another. This function will use the effected variable manifold
        to predict the causative variable manifold. 

        Inputs:
            tau_mode (str): 'mi' or 'acc'
            E_mode (str): 'fnn' or 'simplex'
            'tau_default': 'yes' - means tau is set to one


        Returns:
            true_pred_m (np array): t x E embedded time series, used as the observed dataset to compare with prediction - the causative variable manifold.
            est_pred_m (np array): t x E embedded time series, the predicted manifold - the causative variable manifold prediction. 

        """
    
        #Initialise embedded variables
        self.lib_m, self.pred_m = self.group_embed(tau_mode, E_mode, tau_default).embed_m 
        
        #Do crossmap estimate - use lib manifold to reconstruct and predict pred manifold 
        self.true_pred_m, self.est_pred_m = efn.crossmap(lib_m, pred_m)

        return(self)
        
    #========================
    def conv_cross_map(self, tau_mode, E_mode, tau_default, l_range):
    #========================
    
        """
        This function performs convergent cross mapping between two manifolds: a causative variable (prediction manifold) - one we are testing 
        to see if it causes the other; an effected variable (library manifold) - one we are testing to see if it is caused by the other. CCM 
        is performed over a range of library sizes to check for convergence - the property that if the supposed causative variable actually causes
        the supposed effected variable the correlation between CCM predictions and observed manifold values should increase as more points are 
        added. 

        Inputs:
            tau_mode (str): 'mi' or 'acc'
            E_mode (str): 'fnn' or 'simplex'
            'tau_default': 'yes' - means tau is set to one


        Returns:
            ccm_corr_l (list): list containing CCM correlation values as you increase library 
            ccm_true_l (list): list containing observed prediction manifold as you increase library 
            ccm_pred_l (list): list containing predicted prediction manifold as you increase library 
        """
        #Initialise embedded variables
        lib_m, pred_m = self.group_embed(tau_mode, E_mode, tau_default).embed_m
        
        #Do CCM
        self.ccm_corr_l, self.ccm_true_l, self.ccm_pred_l = efn.CCM_range(l_range, self.cause, self.effect)
        
        return(self)
    