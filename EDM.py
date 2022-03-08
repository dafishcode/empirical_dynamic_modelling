import admin_functions as adfn


#==============================================
#EMBED DATA
#==============================================

#==============================================
def Lorenz(x, y, z, sigma, r, b):
#==============================================

    """
    This function plots the lorenz attractor system, a simplified model of convection rolling - long rolls of counter-rotating air that are oriented approximately parallel to the ground. 
    
    Inputs:
       x (int/float): rate of convective overturning
       y (int/float) horizontal temperature difference
       z (int/float): departure from linear vertical temperature gradient
       s (int/float): prandtl parameter
       r (int/float): rayleigh parameter
       b (int/float): b parameter
    Returns:
       x_d, y_d, z_d (float): values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_d = sigma*(y - x)
    y_d = r*x - y - x*z
    z_d = x*y - b*z
    return x_d, y_d, z_d


#==============================================
def takens_embed(m, tau, data):
#==============================================
    """
    This function takes a singular time series as input and performs lagged coordinate embedding to generate a reconstructed attractor with dimension m and time lag tau. 
    
    Inputs:
        m (int): embedding dimension
        tau (int): time lag into past
        data (np array): 1d vector time series for reconstruction  
    
    Returns:
        data_embed (np array): m x t array of time x embedding dimension
    
    """
    import numpy as np
    data_embed = np.zeros((data.shape[0] - ((m-1)*tau), m))
    
    #loop through each dimension
    for i in range(0, m):
        
        if i == m-1:
            data_embed[:,(m-1)-i] = data[(i*tau):]
        
        else:
            data_embed[:,(m-1)-i] = data[(i*tau):-1* ((m*tau)-((i+1)*tau))]

    return(np.array(data_embed))


#==============================================    
def LE_embed(data, tau):
#==============================================    
    """
    This calculates the lyapunov exponent on an embedded dataset. 
    
    Inputs:
        data (np array): 1d vector timeseries
        tau (int): time lag
        m (int): embedding dimension
        thresh (int): false nn threshold
    
    Returns:
        n_false_NN (int): number of false nns
    
    """


    from sklearn.neighbors import NearestNeighbors 

    le = np.zeros((data.shape[0]-1))
    
    #Find nearest neighbours
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    
    #Loop through each point
    for i in range(1,data.shape[0]-1):
        sum_ = 0
        sum_count = 0
        
        #Loop through each neighbour to i
        for e in range(indices.shape[0]):
            dj0 = np.linalg.norm(data[indices[e][0]] - data[indices[e][1]]) #Distance at time 0
            sep = indices[e][1] - indices[e][0] #Time separation at t0

            #Avoid time points that go past end 
            if e+i < indices.shape[0]:
                d1i_ind = indices[e+i][0]
                d2i_ind = d1i_ind+sep
                if d2i_ind< data.shape[0]:
                    dji = np.linalg.norm(data[d1i_ind] - data[d2i_ind]) #Distance at time i
                    sum_ += np.log(np.abs(dji/dj0))
                    sum_count +=1
             
        if sum_count == 0:
            break
        le[i-1] = (1/ i) *(sum_/sum_count)
                    
    return(le)





#==============================================
#PARAMETER ESTIMATION
#==============================================

#==============================================
def MI(data, delay, n_bins):
#==============================================    
    """
    This function calculates the mutual information of a time series and a delayed version of itself. MI quantifies the amount of information obtained about 1 variable, by observing the other random variable. In terms of entropy, it is the amount of uncertainty remaining about X after Y is known. So we are calculating the amount of uncertainty about time series xi and xi + tau shifted, across a range of taus. To calculate MI for 2 time series, we bin the time series data into n bins and then treat each time point as an observation, and calculate MI using joint probabilities of original time series xi and delayed xi + tau. 
    
    Inputs:
        data (np array): 1d vector timeseries
        delay (int): time lag
        n_bins (int): number of bins to split data into
    
    Returns:
        MI (float): mutual information
    
    """
    
    import math
    import numpy as np
    
    
    MI = 0
    xmax = np.max(data) #Find the max of the time series
    xmin = np.min(data) #Find the min of the time series
    delay_data = data[delay:len(data)] # generate the delayed version of the data - i.e. starting from initial delay
    short_data = data[0:len(data)-delay] #shorted original data so it is same length as delayed data
    size_bin = abs(xmax - xmin) / n_bins #size of each bin
    
    #Define dicts for each probability
    P_bin = {} # probability that data lies in a given bin
    data_bin = {} # data lying in a given bin
    delay_data_bin = {} # delayed data lying in a given bin
    
    
    prob_in_bin = {} #
    condition_bin = {} 
    condition_delay_bin = {}
    
    #Simple function for finding range between values of time series
    def find_range(time_series, xmin, curr_bin, size_bin):
        values_in_range = (time_series >= (xmin + curr_bin*size_bin)) & (time_series < (xmin + (curr_bin+1)*size_bin))
        return(values_in_range)

    #Loop through each bin
    for h in range(0,n_bins):
        
        #calculate probability of a given time bin, unless already defined
        if h not in P_bin:
            data_bin.update({h:  find_range(short_data, xmin, h, size_bin)})
            P_bin.update({h: len(short_data[data_bin[h]]) / len(short_data)})            
            
        #populate probabilities for other time bins 
        for k in range(0,n_bins):
            if k not in P_bin:
                data_bin.update({k: find_range(short_data, xmin, k, size_bin)})
                P_bin.update({k: len(short_data[data_bin[k]]) / len(short_data)})                            
                
            #to calculate the joint probability we need to find the time points where the lagged data lie in a given bin
            if k not in delay_data_bin:
                delay_data_bin.update({k: find_range(delay_data, xmin, k, size_bin) })
                                
            # Find the joint probability, that OG time series lies in bin h and delayed time series lies in bin k
            Phk = len(short_data[data_bin[h] & delay_data_bin[k]]) / len(short_data)

            if Phk != 0 and P_bin[h] != 0 and P_bin[k] != 0:
                MI += Phk * math.log( Phk / (P_bin[h] * P_bin[k]))
    return(MI)

    
#==============================================    
def FNN(data,tau,m, thresh):
#==============================================    
    """
    This function calculates how many nearest neighbours are false neighbours, in an embedded timeseries. Specifically, false nearest neighbours are defined as nearest neighbours to each point in E dimensional embedded space whose distances in E+1 dimensional space are greater than a defined threshold. 
    
    Inputs:
        data (np array): 1d vector timeseries
        tau (int): time lag
        m (int): embedding dimension
        thresh (int): false nn threshold
    
    Returns:
        n_false_NN (int): number of false nns
    
    """
    from sklearn.neighbors import NearestNeighbors 
    import numpy as np

    embed_data_1 = takens_embed(m, tau, data)
    embed_data_2 = takens_embed(m+1, tau, data)
    embed_data_1 = embed_data_1[:embed_data_2.shape[0]] #Shorten embedded time series to match eachother
    
    #Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(embed_data_1)
    distances, indices = nbrs.kneighbors(embed_data_1)

    #two data points are nearest neighbours if their distance is smaller than the standard deviation
    sigma = np.std(distances.flatten())

    n_false_NN = 0
    
    #if distance between points is nonzero, less than the std AND distance between nn in next dimension / previous dimension > threshold = nn is false
    for i in range(embed_data_2.shape[0]):
        if (0 < distances[i,1]) and (distances[i,1] < sigma) and (np.linalg.norm(embed_data_2[i] - embed_data_2[indices[i][1]])/distances[i][1]) > thresh:
            n_false_NN  += 1;
    return n_false_NN 



#====================================
def simplex_project(data, E, tau, t):
#====================================

    """
    This function performs simplex projection over t time steps into the future. Briefly, it splits a time series in library and prediction sets.
    It then embeds the library manifold in E dimensions, using E time lags. For each point p (embedded in E dim space) in the 
    prediction timeseries, the algorithm finds the E+1 nearest neighbours in the library manifold which forms a simplex around 
    the point p. The algorithm then predicts the position of point p at t using the positions of each neighbour at t exponentially weighted by 
    the distances from p at t0. See: Sugihara et al. 'Nonlinear Forecasting as a way of distinguishing chaos from measurement error in time series'.
    
    
    Inputs:
        data (np array): 1d vector of time series to perform simplex projection. 
        E (int): embedding dimension
        tau (int): time delay to use for embedding
        t (int): how many time steps ahead to predict
        
    
    Returns:
        corr (float): correlation coefficient between observed and predicted
        x_tp_m (np array): a vector of observations
        x_tp_pred_m (np array): a vector of predictions
        
    """
    from scipy import spatial
    import numpy as np
    import pandas as pd
    
    
    #==========================================
    def shift_nn(shape, lib_m, dist_mat, nn_ind, nn_ind_tp, curr_dist, nn_num, num, t): 
    #========================================== 
    #This function deals with points that go off manifold at t - finds next nearest neighbours
    
        nn_off = np.sum(nn_ind_tp >= shape) #Number of nearest neighbours that go off manifold at time t
        nn_rem = sorted(range(len(curr_dist)), key=lambda k: curr_dist[k])[nn_num:] #return indeces of ordered remaining nearest neighbours not currently included

        count = 0
        new_nn_tp_l, new_nn_l = [],[]
        for nn in nn_rem: #loop through each remaining neighbour
            if nn + t < shape: #if index of nn + t is on manifold, add to list and count
                new_nn_l = np.append(new_nn_l, nn) #add indeces of new neighbours at t0
                new_nn_tp_l = np.append(new_nn_tp_l, nn+t) #add indeces of new neighbours at tp
                count +=1 
                if count == nn_off: #Stop loop once you have enough neighbours
                    break

        nn_on = nn_ind[nn_ind_tp < shape] #Indeces of OG nearest neighbours at t0 that stay on the manifold into t
        nn_tp_on = nn_ind_tp[nn_ind_tp < shape] #Indeces of OG nearest neighbours at t that stay on the manifold at t

        new_nn_ind = (np.append(nn_on, new_nn_l)).astype(int) #add nearest neighbour points at t0 that stay on manifold up to tp, to new points
        new_nn_ind_tp = (np.append(nn_tp_on, new_nn_tp_l)).astype(int) #add nearest neighbour points at tp that stay on manifold, to new points


        nn = lib_m[new_nn_ind] #positions of nearest neighbours in library, to current point in pred at t0
        nn_dist = dist_mat[num, new_nn_ind]  #distances of each nn to our pred point
        w_mat = np.exp(-1*(nn_dist/np.min(nn_dist))) #matrix of weights for each nn

        return(new_nn_ind_tp, w_mat)
    

    # split data in half into library and prediction
    lib = data[:data.shape[0]//2]
    pred = data[data.shape[0]//2:]

    # Build manifold with given E and tau
    lib_m = takens_embed(E, tau, lib)
    pred_m = takens_embed(E, tau, pred)

    x_tp_m = np.zeros(pred_m.shape[0]) #Matrix to enter values you are trying to predict
    x_tp_pred_m = np.zeros(pred_m.shape[0]) #Matrix to values you have predicted
    x_tp_m[:] = np.nan #Make all nan to deal with empty values
    x_tp_pred_m[:] = np.nan


    #find the E+1 nearest neighbours in library
    dist_mat = spatial.distance_matrix(pred_m, lib_m) #compute distances between all points
    nn_num = E+1 #how many nearest neighbours to find


    #Loop through each point in pred
    for num in range(pred_m.shape[0]-t):

        # Find nearest neighbours in library for each pred_m point
        current_point = pred_m[num]
        curr_dist = dist_mat[num]
        nn_ind = sorted(range(len(curr_dist)), key=lambda k: curr_dist[k])[:nn_num] #return indeces of nearest neighbours in library

        #Calculate weights for simplex projection - weights are calculated from nn distance at t0
        nn = lib_m[nn_ind] #positions of nearest neighbours in library, to current point in pred at t0
        nn_dist = dist_mat[num, nn_ind]  #distances of each nn to our pred point
        w_mat = np.exp(-1*(nn_dist/np.min(nn_dist))) #matrix of weights for each nn 

        # Where do nn end up at t + n
        nn_ind_tp = np.array(nn_ind) + t #find indeces of neighbours in the future for simplex projection

        #Deal with points that go off manifold at t - find next nearest neighbours 
        if sum(nn_ind_tp >= lib_m.shape[0]) >0:                
            #Replace neighbours that go off manifold at t with neighbours that dont, and recalculate weights
            nn_ind_tp, w_mat = shift_nn(lib_m.shape[0], lib_m, dist_mat, np.array(nn_ind), nn_ind_tp, curr_dist, nn_num, num, t)

        nn_tp = lib_m[nn_ind_tp] # locations of neighbours in future

        #Simplex project - how much do the positions of neighbours relative to point of interest change over time 
        #use weights from t 0
        #use neighbour points from t + n
        x_tp = pred_m[num][0] #Point I am trying to predict 
        x_tp_pred = 0
        for nn_i in range(w_mat.shape[0]): #Loop through each nn and sum over the weight*position at tp
            x_tp_pred+= (w_mat[nn_i]/np.sum(w_mat))*nn_tp[nn_i]
        x_tp_pred = x_tp_pred[0] #project back into 1d space

        x_tp_m[num] = x_tp #true 
        x_tp_pred_m[num+t] = x_tp_pred  #estimated - NB you are estimating the future value at t, not the original
        

    my = {'Obs': x_tp_m, 'Pred': x_tp_pred_m}
    my_df = pd.DataFrame(data=my) 
    corr = my_df.corr()['Obs']['Pred']
        
        
    return(corr, [x_tp_m, x_tp_pred_m])

#==============================================    
def find_tau(data, mode):
#==============================================    
    """
    This function estimates tau for lagged coordinate embedding, using different approaches. mi = find the tau that provides the first minima of the MI - this provides most independent information to initial time series without completely losing the time series. ac = find the tau at which the autocorrelation drops below 1/e. 
    
    Inputs:
        data (np array): 1d vector timeseries
        mode (str): 'mi' or 'ac'
    
    Returns:
        tau (int): estimated tau for embedding
    
    """
    
    if mode == 'mi':
    
        import numpy as np
        from scipy.signal import argrelextrema

        MI_list = []
        for i in range(1,50):
            MI_list = np.append(MI_list,[MI(data,i,50)])

        tau = argrelextrema(MI_list, np.less)[0][0] + 1 #find the first minima of MI function
        return(tau)
    
    if mode == 'ac':
        import numpy as np
        
        auto = adfn.autocorr(data, 50)
        tau = np.min(np.where(auto < 1/np.e)) #find the tau at which the autocorrelation drops below 1/e
        return(tau)


#==============================================    
def find_E(data, tau, mode):
#==============================================    
    """
    This function estimates the embedding dimension E for lagged coordinate embedding, using different approaches. 
    fnn = find the E that approaches 0 false nearest neighbours - what embedding unfolds the manifold so that nearest neighbours become preserved.
    simplex = runs simplex projection over a range of E values with a given tau, and returns the E with greatest correlation between the real variable and predicted. 
    
    Inputs:
        data (np array): 1d vector timeseries
        tau (int): delay for embedding
        mode (str): 'fnn' or 'simplex'
    
    Returns:
        E (int): estimated number of dimensions to use for embedding
    
    """
    
    if mode == 'fnn':
        import numpy as np
        from scipy.signal import argrelextrema

        nFNN = []
        for i in range(1,15):
            nFNN.append(FNN(data,tau,i, 10) / len(data))

        E = np.min(np.where(np.array(nFNN) ==0 )) + 1
        return(E)
    
    if mode == 'simplex':
        import numpy as np
        
        E_range = 20 
        t = 1 
        
        corr_l = [0]*E_range
        for E in range(1, E_range+1):
            corr_l[E-1] = simplex_project(data, E, tau, t)[0]
            #print('Done E = ' + str(E))

        E_max = np.where(corr_l == np.max(corr_l))[0][0] + 1
        return(E_max)


    
    
#==============================================
#Cross mapping
#==============================================

#====================================
def crossmap(lib_m, pred_m):
#====================================
   
    """
    This function performs cross map predictions from one manifold to another. Briefly, the algorithm takes two different manifold and uses on to predict
    the other - if manifold Y can accurately predict manifold X, then Y contains information about X within it and thus X must cause Y. For each point in 
    manifold X, we find the nearest neighbours to that point and then locate the same nearest neighbours (labelled by their time indeces) on manifold Y. We 
    then use the locations of nn in Y and the distances between point of interest p on X and its nearest neighbours in X, to predict where point p ends up in Y. 
    The prediction will be accurate if the local structure of the manifold is converved across manifold X and Y - i.e. nearest neighbours to p on X are also nearest
    neighbours to p on Y. 
    
    
    Inputs:
        lib_m (np array): t x E embedded time series, used to make the prediction.
        pred_m (np array): t x E embedded time series, used as the observed dataset to compare with prediction. 
        
    
    Returns:
        x_m (np array): t x E embedded time series, used as the observed dataset to compare with prediction.
        x_pred_m (np array): t x E embedded time series, the predicted manifold. 
        
    """
    import numpy as np
    from scipy import spatial
    
    #make sure each manifold is the same length
    mini = np.min([pred_m.shape[0], lib_m.shape[0]])
    pred_m = pred_m[:mini,:]
    lib_m = lib_m[:mini,:]
    
    x_m = np.zeros(pred_m.shape[0]) #Matrix to enter values you are trying to predict
    x_pred_m = np.zeros(pred_m.shape[0]) #Matrix to values you have predicted
    x_m[:], x_pred_m[:] = np.nan, np.nan #Make all nan to deal with empty values

    #find the E+1 nearest neighbours in library
    dist_mat = spatial.distance_matrix(lib_m, lib_m) #compute distances between all points against themselves
    nn_num = lib_m.shape[1]+1 #how many nearest neighbours to find
    
    #Loop through each time step in lib
    for t in range(lib_m.shape[0]):
        # Find nearest neighbours in library for current point in library
        current_point = lib_m[t]
        curr_dist = dist_mat[t]
        nn_ind = sorted(range(len(curr_dist)), key=lambda k: curr_dist[k])[:nn_num+1][1:] #return indeces 

        nn = lib_m[nn_ind] #positions of nearest neighbours in library, to current point in lib
        nn_pred = pred_m[nn_ind] #positions of points in pred, labelled by indeces of nearest neighbours in lib to point in lib

        #Reconstruct pred point
        #Use weights calculated from distances between lib point and its nearest neighbours in lib
        #Use coordinates of pred points sharing time indeces with lib nearest neighbours

        #CALCULATE WEIGHTS
        nn_dist = dist_mat[t, nn_ind]  #distances of each nn to our pred point
        w_mat = np.exp(-1*(nn_dist/np.min(nn_dist))) #matrix of weights for each nn 

        #SUM OVER ALL PRED POINTS
        x_ = pred_m[t][0] # Value I am trying to predict
        x_pred = 0 # Predicted value
        for nn_i in range(w_mat.shape[0]): #Loop through each nn in lib and sum over the weight*position in pred
            x_pred+= (w_mat[nn_i]/np.sum(w_mat))*nn_pred[nn_i]
        x_pred = x_pred[0] #project back into 1d space

        #Populate vectors
        x_m[t] = x_
        x_pred_m[t] = x_pred
        
    return(x_m, x_pred_m)


#==============================================
def CCM_range(l_range, cause, effect):
#==============================================
    
    """
    This function performs convergent cross mapping between two manifolds: a causative variable (prediction manifold) - one we are testing 
    to see if it causes the other; an effected variable (library manifold) - one we are testing to see if it is caused by the other. CCM 
    is performed over a range of library sizes to check for convergence - the property that if the supposed causative variable actually causes
    the supposed effected variable the correlation between CCM predictions and observed manifold values should increase as more points are 
    added. 
    
    Inputs:
        l_range (np array): 1d vector of library sizes to test CCM
        cause (dict): dictionary for the causative variable, containing the data and parameters
        effect (dict): dictionary for the effected variable, containing the data and parameters
    
    Returns:
        corr_l (list): list containing CCM correlation values as you increase library 
        true_l (list): list containing observed prediction manifold as you increase library 
        pred_l (list): list containing predicted prediction manifold as you increase library 
    
    """    

    import random
    from scipy import stats
    import numpy as np
    
    lib, lib_E, lib_tau = effect['data'], effect['E'], effect['tau'] # This is the variable that will be used to predict -  the effected variable.
    pred, pred_E, pred_tau = cause['data'], cause['E'], cause['tau'] # This is the variable that will be predicted - the causative variable. 

    #Embed data
    lib_m = takens_embed(lib_E, lib_tau, lib) 
    pred_m = takens_embed(pred_E, pred_tau, pred) 

    #Initialise data output structures
    true_l, pred_l = [0]*len(l_range), [0]*len(l_range)
    corr_l = [0]*len(l_range)

    smallest = np.min([lib_m.shape[0], pred_m.shape[0]]) #find smallest array - may be different sizes if tau and E are different
    
    #Cross map as you increase library size
    for e,l in enumerate(l_range):
        t_l = random.sample(range(smallest),l) #Randomly sample
        lib_m_sub, pred_m_sub = lib_m[t_l], pred_m[t_l]
        true, pred = crossmap(lib_m_sub, pred_m_sub) #Run cross map on subsampled data
        true_l[e], pred_l[e] = true,pred
        corr_l[e] = stats.pearsonr(true, pred)[0]
    return(corr_l, true_l, pred_l)







