#SORT
#=============================
#=============================

#=========================================
def return_files(path, experiment, search):
#=========================================

    """
    Return list of files in defined path, and changes current working directory to path containing desired files
    
    Inputs:
    path (string): name of parent path
    experiment (string): name of experiment folder 
    search (string): words that files must contain
    
    Returns:
    data_list (list): list containing all files of interest
    
    """

    import os
    import glob
    os.chdir(path + experiment)
    data_list = sorted(glob.glob(search))
    return(data_list)


#=============================
def name_zero(pad, start, end, step): 
#=============================
    """
    Adds zero to front or back of a list of numbers - useful for saving filenames with numbers
    
    Inputs:
    pad (string): 'front' or 'back'
    start (int): number to start from
    end (int): number to end wtih
    step (int): stepsize
    
    Returns:
    listme (list): a list of strings with 0s appended
    
    """
    
    
    import os 
    import numpy as np
    
    if pad == 'front': 
        count = 0
        listme = list(range(start, end+1, step))
        for i in range(start, end+1, step):
            if i < 10:
                num = '0' + str(i) #add 0 to front if integer value less than 10
            elif i >9:
                num = str(i) #else do not add 0
            listme[count] = num
            count+=1
        return(listme)

    if pad == 'back': 
        count, count1 = 0,0
        looplist = np.arange(start, end + step, step)
        listme = list(range(0, looplist.shape[0]))
        lenlist = list(range(looplist.shape[0]))
        for i in looplist:
            lenlist[count1] = len(str(round(i, len(str(step)))))
            count1 +=1
        for i in looplist:
            if len(str(round(i,len(str(step))))) < np.max(lenlist):
                num = str(round(i,len(str(step)))) + '0'
            else:
                num = str(round(i,len(str(step))))
            listme[count] = num
            count+=1
        return(listme)

#=============================
def repeat_list(name, length): 
#=============================  
    """
    Creates list of the same string repeated n times
    
    Inputs:
    name (string): string to repeat
    length (int): length of list
    
    Returns:
    itlist (list): a list of repeated string
    
    """
    
    itlist = list(range(length))
    for i in range(len(itlist)):
        itlist[i] = name
    return(itlist)


#==============================
def save_name(name): 
#===============================
    """
    Creates name template for saving - requires standardised input format
    
    Inputs:
    name (string): full input string
    
    Returns:
    (string): name template
    
    """

    return(name[:name.find('run')+6])


#PROCESS
#=============================
#==============================

#============================================
def save_shared_files(path, son_path, mode):
#============================================
    """
    Saves shared modules across different repositories
    
    Inputs:
    path (string): name of parent path
    son_path (string): name of code folder 
    mode (string): define which file to save: 'admin' or 'criticality'
    
    """

    import os
    from shutil import copyfile
    
    def loop_dir(file_list, path_list):
        """
        Loop between directories and save file in all but current 

        """

        #Loop through files and directories
        for x in file_list:
            for e,i in enumerate(path_list):
                if path_list[e] not in os. getcwd() and not path_list[e].startswith('.'): #skip current directory
                    copyfile(x, path + i + os.sep + x) #copy and overwrite files in directory

    if mode == 'admin':
        file_list = return_files(path , son_path, 'admin_functions.py' ) #search for admin file in current directory
        path_list = os.listdir(path) #get names of all directories


    elif mode == 'criticality':
        file_list = [return_files(path , son_path, 'avalanches.py')[0], return_files(path , son_path, 'IS.py')[0]]  #search for admin file in current directory
        path_list = ['criticality', 'spiking_network_criticality', 'zebrafish_mutant_analysis'] #CHANGE AS NEEDED!

    loop_dir(file_list, path_list) 


#=====================================================================
def parallel_func(cores, savepath, iter_list, func, param_list, name, variables, mode): 
#=====================================================================
    """
    This function allows parallel pooling of processes using functions
    
    Inputs:
    cores = number of cores 
    savepath = path for saving
    iter_list = list with parameter inputs that you will parallel process (inputs must be at start of function)
    func = function name
    param_list = list containing remaining function parameters 
    name = filename for saving, should be unique if mode = save_group
    variables = list containing name endings for each variable, if function returns multiple
    mode = output type:
        save_single - saves each variable of function output individually
        save_group - saves all batched function outputs in a list
        NA - returns all batched function outputs in a list, without saving
    """
    
    from multiprocessing import Pool
    import numpy as np
    pool = Pool(cores) #number of cores
    count = 0

    batch_list = list(range((np.int(len(iter_list)/cores)))) #define number of batches

    for i in range(len(batch_list)): #process each batch
        cores_inputs = list(range(cores)) #define input for each core
        for e in range(len(cores_inputs)):  
            sub_iter_list = iter_list[count:count+1] #Find current iter value - add to subset iter_list
            sub_iter_list.extend(param_list) #Append current iter value onto remaining parameter
            cores_inputs[e] = sub_iter_list 
            count+=1
        batch_list[i] = pool.starmap(func, cores_inputs) #pool process on each core

        if mode == 'save_single':
            for t in range(cores):  #loop through each core in current loop
                for f in range(len(batch_list[i][t])):
                    save_var = batch_list[i][t][f] #function output for current core in current batch
                    save_name = name + '-' + str(cores_inputs[t][0]) + '-' + variables #save name based on iterable parameter
                    np.save(savepath + save_name, save_var)
        
    if mode != 'save_single':
        #Append all calculated value together
        if isinstance(batch_list[0][0], int) or isinstance(batch_list[0][0], float) :
            return_me = np.hstack(np.array(batch_list))
        else:
            return_list = list(range(len(batch_list[0][0])))
            new_array = np.vstack(np.array(batch_list))
            return_me = [new_array[:,i] for i in range(new_array.shape[1])]

        if mode == 'save_group':
            save_name = name
            np.save(savepath + save_name, return_me)

        else:
            return(return_me)
      
    
#=====================================================================
def parallel_class(cores, savepath, iter_list, func, param_list, name, variables, mode): 
#=====================================================================
    """This function allows parallel pooling of processes using classes
    
    Inputs:
    cores = number of cores 
    savepath = path for saving
    iter_list = list with parameter inputs that you will parallel process (inputs must be at start of function)
    func = function name
    param_list = list containing remaining function parameters 
    name = filename for saving, should be unique if mode = save_group
    variables = list containing name endings for each variable, if function returns multiple
    mode = output type:
        save_single - saves each variable of function output individually
        save_group - saves all batched function outputs in a list
        NA - returns all batched function outputs in a list, without saving
    """
    
    from multiprocessing import Pool
    import numpy as np
    pool = Pool(cores) #number of cores
    count = 0

    batch_list = list(range((np.int(len(iter_list)/cores)))) #define number of batches
    for i in range(len(batch_list)): #process each batch
        cores_inputs = list(range(cores)) #define input for each core
        for e in range(len(cores_inputs)):  
            sub_iter_list = iter_list[count:count+1] #Find current iter value - add to subset iter_list
            sub_iter_list.extend(param_list) #Append current iter value onto remaining parameter
            cores_inputs[e] = sub_iter_list 
            count+=1
        batch_list[i] = pool.starmap(func, cores_inputs) #pool process on each core
        
        

        
        if mode == 'save_single':
            import scipy.sparse
            for t in range(cores):  #loop through each core in current loop
                for s in range(len(variables)):
                    save_var = batch_list[i][t].__dict__[variables[s]] #function output for current core in current batch
                    save_name = name + '-' + str(cores_inputs[t][0]) + '-' + variables[s] #save name based on iterable parameter
                    sparse_matrix = scipy.sparse.csc_matrix(save_var)
                    scipy.sparse.save_npz(savepath + save_name, sparse_matrix)
                    #np.save(savepath + save_name, save_var)

    if mode != 'save_single':
    
        #Append all calculated values together
        if len(variables) == 1:
            if isinstance(batch_list[0][0].__dict__[variables[0]], int) or isinstance(batch_list[0][0].__dict__[variables[0]], float):
                count=0
                return_me = list(range(len(iter_list)))
                for first in range(len(batch_list)):
                    for second in range(len(batch_list[0])):
                        return_me[count] = batch_list[first][second].__dict__[variables[0]]
                        count+=1
            else:            
                count=0
                return_me = list_of_list(len(variables),len(iter_list))
                for first in range(len(batch_list)):
                    for second in range(len(batch_list[0])):
                        for third in range(len(variables)):
                            return_me[third][count] = batch_list[first][second].__dict__[variables[third]]
                        count+=1       

        if len(variables) > 1:
            count=0
            return_me = list_of_list(len(variables),len(iter_list))
            for first in range(len(batch_list)):
                for second in range(len(batch_list[0])):
                    for third in range(len(variables)):
                        return_me[third][count] = batch_list[first][second].__dict__[variables[third]]
                    count+=1
        
        if mode == 'save_group':
            save_name = name
            np.save(savepath + save_name, return_me)
        
        else:
            return(return_me)
    
        
#=======================================================================================        
def timeprint(per, r, numrows, name):
#=======================================================================================
    """ 
    Print current time step every percentile
    
    Inputs:
    per = how often you want to print (as percentiles)
    r = current iterator value
    numrows = total number of steps
    name = name to output
    """
    if r % round((per*numrows/100)) == 0: 
            print("Doing number " + str(r) + " of " + str(numrows) + " for " + name)
            
            
#MATHS
#=============================
#=============================

#=======================================================================================
def window(size, times): #make window of given size that is divisible by time series
#=======================================================================================
    """
    Returns the window size that is the closest divisor of a timeseries to given input
    
    Inputs:
    size (int): ideal window size
    times(int): overall trace length
    
    Returns: 
    size (int): window size that is divisible by trace (rounds up)
    n_windows (int): number of windows that split up trace
    """
    for i in range(times):
        if times % size ==0:
            break
        else:
            size+=1
    n_windows = int(times/size)
    return(size, n_windows)

