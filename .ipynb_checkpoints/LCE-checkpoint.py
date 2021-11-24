#£Check

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
        tau (int): time lag
        data (np array): 1d vector time series for reconstruction  
    
    Returns:
        data_embed (np array): m x t array of time x embedding dimension
    
    """
    import numpy as np
    
    data_embed = np.zeros((data.shape[0] - (m*tau), m))
    for i in range(0, m):
        data_embed[:,i] = data[(i*tau):-1* ((m*tau)-(i*tau))]
    return(data_embed)







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
                MI += -Phk * math.log( Phk / (P_bin[h] * P_bin[k]))
    return(MI)


#==============================================    
def FNN(data,tau,m, thresh):
#==============================================    
    """
    This function performs the false nearest neighbours algorithm to identify the correct embedding dimension. 
    
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
    
    embed_data = takens_embed(m, tau, data)
    
    # find nearest neighbours 
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(embed_data)
    distances, indices = nbrs.kneighbors(embed_data)
    
    #two data points are nearest neighbours if their distance is smaller than the standard deviation
    sigma = np.std(distances.flatten())
    
    n_false_NN = 0
    
    
    for i in range(0, len(data)-tau*(m+1)):
        
        #if distance between points is nonzero, less than the std AND distance between nn in next dimension / previous dimension > threshold = nn is false
        if (0 < distances[i,1]) and (distances[i,1] < sigma) and ( (abs(data[i+m*tau] - data[indices[i,1]+m*tau]) / distances[i,1]) > thresh):
            n_false_NN  += 1;
    return n_false_NN 



#==============================================    
def find_taum(data, thresh):
#==============================================    
    """
    This function finds the tau that maximises MI and m that approaches 0 false nearest neighbours.  
    
    Inputs:
        data (np array): 1d vector timeseries
    
    Returns:
        tau (int): time delay that maximises MI
        m (int): embedding dimension that first approaches 0 false nearest neighbours
    
    """
    import numpy as np
    
    MI_list = []
    for i in range(1,21):
        MI_list = np.append(MI_list,[MI(data,i,50)])

    tau = np.where(MI_list == np.min(MI_list))[0][0] + 1

    nFNN = []
    for i in range(1,15):
        nFNN.append(FNN(data,tau,i, thresh) / len(data))

    m = np.min(np.where(np.array(nFNN) ==0 )) + 1
    return(tau,m)




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