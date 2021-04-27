import admin_functions as adfn

def create_timeseries(name, coord, fulltrace, fulldff, block_len, pref_list, savepath):
    import numpy as np
    trace = fulltrace
    dff = fulldff
    np.save(savepath + name + '_coord.npy', coord)
    np.save(savepath + name + '_trace.npy', trace)
    np.save(savepath + name + '_deltaff.npy', dff)

    #brainsum
    mean = np.apply_along_axis(np.mean, 0, trace)
    np.save(savepath + name + '_brainsum.npy', mean)

    window = adfn.window(50, mean.shape[0])[0]
    scalar = 3
    percentile = 0.08
    gen_index = mean_si(mean, window, scalar, percentile)
    np.save(savepath + name + '_si.npy', gen_index)

    start,stop = 0,block_len
    for x in range(len(pref_list)):
        #Split trace
        split_trace = fulltrace[:,start:stop]
        split_dff = fulldff[:,start:stop]
        start+=block_len
        stop+=block_len
        np.save(savepath + name + '_' + pref_list[x] +  '_trace.npy', split_trace)
        np.save(savepath + name + '_' + pref_list[x] + '_deltaff.npy', split_dff)

        #brainsum
        split_mean = np.apply_along_axis(np.mean, 0, split_trace)
        np.save(savepath + name + '_' + pref_list[x] + '_brainsum.npy', split_mean)
        split_gen_index = mean_si(split_mean, window, scalar, percentile)
        np.save(savepath + name + '_' + pref_list[x] + '_si.npy', split_gen_index)
    
    
    
def mean_si(mean, window, scalar, percentile):
    import numpy as np
    baseline = np.zeros(mean.shape[0])
    for i in range(mean.shape[0]):
        baseline[i] = (np.mean(mean[np.where(mean < np.quantile(mean, percentile, axis=0))]))

    meanbase = np.mean(baseline)
    gen_index = []
    for e in range(mean.shape[0]):
        if np.mean(mean[e:e+window]) > scalar*meanbase:
            gen_index = np.append(gen_index, e)
    return(gen_index)

def kmeans_traces(name, n_clus, fulltrace, block_len, pref_list, savepath):
    import numpy as np
    from sklearn.cluster import KMeans

    #whole trace
    kmeans   = KMeans(n_clus, random_state=0).fit(fulltrace)  #perform k means on all cells
    klab =  kmeans.labels_
    np.save(savepath + name + '_kmeans.npy', klab)

    start,stop = 0,block_len
    for x in range(len(pref_list)):
        #Split trace
        split_trace = fulltrace[:,start:stop]
        kmeans   = KMeans(n_clus, random_state=0).fit(split_trace)  #perform k means on all cells
        klab =  kmeans.labels_
        start+=block_len
        stop+=block_len
        np.save(savepath + name + '_' + pref_list[x] +  '_kmeans.npy', klab)