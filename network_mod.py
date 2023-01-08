#=====================
#=====================
class ws_netsim: 
#=====================
#=====================
    """
    Class to build watts-strogatz networks and run avalanche simulations
    dist = distance matrix between all nodes in network
    
    """

    #========================
    def __init__(self,dist):
    #========================
        import numpy as np
        self.dist = dist
    

    #BUILD NETWORK
    #=================
    #=================
    #====================================
    def k_neighbours(self, edge_density, mode):
    #====================================
        import numpy as np
        """
        Form connections with k-nearest neighbours
        Inputs: 
            edge_density = number of neighbours to connect to
            mode = directed or undirected
        """
        
        # Loop through rows of distance matrix to find k_neighbours
        #-----------------------------------------------------------------------------
        for row in range(self.dist.shape[0]):
            k_neighbours = int(edge_density) #Find k_neighbours for each cell
            neighbours = self.dist[row,].argsort()[:k_neighbours+1][::-1] #find neighbours 
            self.A[row,neighbours[:-1]] = 1 #make all edges of neighbours connected in network
            
            if mode == 'undirected':
                self.A[neighbours[:-1],row] = 1
        return(self)

    #=====================================
    def net_generate(self, edge_density, p, mode):
    #=====================================
        """
        Generate random small world graph with specific Edge density. The Watts-Strogatz model has (i) a small average shortest path length, and (ii) a large clustering coefficient. The algorithm works by assigning a pre-defined number of connections between k-nearest neighbours - it then loops through each node and according to some uniform probability re-assigns its edges from its connected k-nearest neighbours and a random unconnected node. 
        Inputs:
            edge_density = number of k_nearest neighbours each node is connected to
            p = probability of an edge being randomly re-assigned
            mode = directed or undirected
        """
        import numpy as np
        import networkx as nx
        import random
        import copy
        
        if mode!= 'directed' and mode!= 'undirected': 
            print('Select directed or undirected')
            exit()
        self.A = np.zeros(self.dist.shape)
        self.k_neighbours(edge_density, mode)

        # Rewire connections with certain probability
        #-----------------------------------------------------------------------------
        
        if mode == 'undirected':
            [rows, cols]    = np.where(np.triu(self.A) == 1) 
            probs           = np.random.uniform(size = rows.shape[0]) #Generate random values for each connection 
            edges_to_change = np.where(probs <= p)[0] #see which values are randomly changed
            
            for e in range(edges_to_change.shape[0]): #Loop through edges to change
                this_edge = edges_to_change[e]
                self.A[rows[this_edge], cols[this_edge]] = 0         # switch off old edge
                self.A[cols[this_edge], rows[this_edge]] = 0

                where_0 = np.where(self.A[rows[this_edge]] == 0)[0] #find possible connections to reassign to
                new_edge = random.choice(where_0[np.where(where_0 !=rows[this_edge])[0]]) #randomly choose one - ignoring any connections on the diagonal 
                #Assign connection
                self.A[rows[this_edge], new_edge] = 1        # switch on new edge
                self.A[new_edge, rows[this_edge]] = 1
        
        if mode == 'directed':
            [rows, cols]    = np.where(self.A == 1) 
            probs           = np.random.uniform(size = rows.shape[0]) #Generate random values for each connection 
            edges_to_change = np.where(probs <= p)[0] #see which values are randomly changed
        
            # Rewire connections with certain probability
            #-----------------------------------------------------------------------------
            [rows, cols]    = np.where(self.A == 1) 
            probs           = np.random.uniform(size = rows.shape[0]) #Generate random values for each connection 
            edges_to_change = np.where(probs <= p)[0] #see which values are randomly changed

            for e in range(edges_to_change.shape[0]): #Loop through edges to change
                this_edge = edges_to_change[e]
                self.A[rows[this_edge], cols[this_edge]] = 0         # switch off old edge

                where_0 = np.where(self.A[rows[this_edge]] == 0)[0] #find possible connections to reassign to
                new_edge = random.choice(where_0[np.where(where_0 !=rows[this_edge])[0]]) #randomly choose one - ignoring any connections on the diagonal 
                #Assign connection
                self.A[rows[this_edge], new_edge] = 1        # switch on new edge
        return(self)

    
    #CALCULATE CYCLES
    #=================
    #=================
    #===========================
    def cycles_calculate(self, edge_density, p, mode):
    #===========================
        import networkx as nx
        import numpy as np
        
        cyc_mat = self.net_generate(edge_density, p, mode).A #matrix to calculate cycles
        G = nx.from_numpy_matrix(cyc_mat)
        cyc = nx.algorithms.cycle_basis(G)
        edge =  int(np.sum(cyc_mat))
        self.cycles = len(cyc)
        self.edges = edge
        return(self)
        
    #===========================
    def cycles_median(self, edge_density, p, n_samp, mode):
    #===========================
    #select median cycles number for simulations - ensure you capture non-skewed cycle values
        import networkx as nx
        import numpy as np
        cyc_list = list(range(n_samp)) #list containing cycle densities for each iteration
        cyc_mat_list = list(range(n_samp)) #list containing each generated matrix
        for i in range(n_samp):
            curr_mat = self.net_generate(edge_density, p, mode).A #matrix to calculate cycles
            G = nx.from_numpy_matrix(curr_mat)
            cyc = nx.algorithms.cycle_basis(G)
            edge =  int(np.sum(curr_mat))
            cyc_mat_list[i] = curr_mat
            cyc_list[i] = len(cyc)/edge
        if n_samp % 2 == 0:
            self.sim_A  = cyc_mat_list[min(range(len(cyc_list)), key=lambda x: abs(cyc_list[x]-np.median(cyc_list)))] #matrix to run simulation on

        else:
            self.sim_A  = cyc_mat_list[np.where(cyc_list == np.median(cyc_list))[0][0]] #matrix to run simulation on

        return(self) 
    
    
    #BUILD WEIGHT MATRIX
    #===================
    #===================
    # Simple sigmoid function to 'soften' the exponential
    #===========================
    def sig(self, x):
    #===========================
        import numpy as np
        self.sig_output = 1 / (1+np.exp(-x))
        return(self)
    
    # Conversion from distance to edge weights, scaled (itself exponentially) by s
    #====================================
    def dist2edge(self, distance, divisor, soften, s):
    #===================================
        import numpy as np
        self.edge_weight_out = np.exp(s/5)*self.sig(np.exp(-soften/np.exp(s)*distance)).sig_output/divisor
        return(self)  
    
    #===========================
    def adjmat_generate(self, s, edge_density, p, n_samp, divisor, soften, mode):
    #===========================
        import numpy as np
        import copy
        mat = np.zeros((self.dist.shape))
    
        curr_mat = self.cycles_median(edge_density, p, n_samp, mode).sim_A
        
        [rows, cols]    = np.where(np.triu(curr_mat) == 1) 
        for e in range(len(rows)):
            edge_weight = self.dist2edge(self.dist[rows[e], cols[e]], divisor, soften, s).edge_weight_out
            mat[rows[e], cols[e]] = edge_weight 
            mat[cols[e], rows[e]] = edge_weight
        self.adj_mat = copy.deepcopy(mat)
            
        return(self)
    

    
    #SIMULATE AVALANCHES
    #===================
    #===================
    
    #Find cells to propagate
    #=====================================================
    def propagate_neighbours(self, curr_mat, start_node):
    #=====================================================
        import numpy as np
        self.prop_nodes = []
        nodes = np.where(curr_mat[start_node] > 0) [0]
        weights = curr_mat[start_node][nodes]
        for f in range(len(nodes)):
            if weights[f] > np.random.uniform(0, 1):
                self.prop_nodes = np.append(self.prop_nodes, nodes[f])
        return(self)

    
    #Simulate 
    #===========================
    def simulate(self,  s, edge_density, p, n_samp, divisor, soften, cutoff, n_sims, mode):
    #===========================
        """
        Simulate output size for a given input

            Inputs:
                edge_density = number of k_nearest neighbours each node is connected to
                p = probability of an edge being randomly re-assigned
                n_samp = number of networks to generate when calculating median cycle density
                divisor = divisor value for weight scaling function
                soften = degree of exponential softening for weight scaling function
                cutoff = when to stop an avalanche
                n_sims = number of simulations
                mode = directed or undirected

        """
    
        import numpy as np
        curr_mat = self.adjmat_generate(s, edge_density, p, n_samp, divisor, soften, mode).adj_mat

        self.av_size = []
        self.av_dur = []

        for i in range(n_sims):
            #Decide start node
            start_node = np.random.uniform(0, curr_mat.shape[0]-1)
            down = int(start_node)
            up= int(start_node)+1
            if np.random.uniform(down, up) >= start_node:
                start_node = up
            else:
                start_node = down


            #Initialise avalanche - ping first node
            t_nodes = self.propagate_neighbours(curr_mat, start_node).prop_nodes #Find connected neighbours > threshold
            curr_list = t_nodes
            iterate = 'yes'

            if len(t_nodes) > 1: #must have at least 3 cells to begin avalanche
                all_nodes = np.append(start_node, t_nodes)
                timesteps = 1

                while iterate == 'yes':
                    tplus_nodes = []
                    for z in range(len(curr_list)):
                        #List of all nodes active in next timestep
                        tplus_nodes = np.append(tplus_nodes, self.propagate_neighbours(curr_mat, int(curr_list[z])).prop_nodes)

                    all_nodes = np.append(all_nodes, tplus_nodes)
                    timesteps+=1
                    curr_list = tplus_nodes

                    if len(all_nodes) > cutoff:
                        iterate = 'no'

                    if len(tplus_nodes) == 0: #if no more active cells - stop
                        iterate = 'no'


                self.av_size = np.append(self.av_size, len(all_nodes)) 
                self.av_dur = np.append(self.av_dur, timesteps)

            else:
                continue

        return(self)
    
    
    
    
    
#=====================
#=====================
class ba_netsim: 
#=====================
#=====================
    """
    Class to build barabasi-albert networks and run avalanche simulations
    dist = distance matrix between all nodes in network
    """

    #========================
    def __init__(self,dist):
    #========================
        import numpy as np
        self.dist = dist
    

    #BUILD NETWORK
    #=================
    #=================
    
    #=====================================
    def sample(self, seq, m):
    #=====================================
        """ Return m unique elements from seq.
        """
        import random
        import numpy as np
        
        #make targets a set - only contains unique elements
        targets=set()
        while len(targets)<m:
            x=random.choice(seq)
            targets.add(x) #add method only adds if x is not already in target set
        return np.array(list(targets))
    
    #=====================================
    def connect(self, edge_density, add_list):
    #=====================================
        current_n = edge_density #current number of nodes

        # Nodes to connect to from current node
        nodes_out =list(range(edge_density))

        # Sequence of all nodes connected (in and out) - can sample from this 
        node_counts=[]

        #iterate until number of nodes = n
        while current_n < self.dist.shape[0]:
            listlist = [current_n, nodes_out]
            for t in range(len(add_list)):
                self.A[listlist[add_list[t][0]],listlist[add_list[t][1]]] = 1

            #add current nodes receiving outgoing connections to node sequence
            node_counts.extend(nodes_out)

            #list of incoming connections for current node - i.e. repeated sequence of current node
            nodes_in = [current_n]*edge_density

            #add current node (as many times as it sends out connections - assumes undirected network) to node sequence
            node_counts.extend(nodes_in)

            #update nodes_out - uniformly sample from sequence of node_counts
            nodes_out = self.sample(node_counts, edge_density)

            current_n +=1
        return(self)
    
    #=====================================
    def net_generate(self, edge_density, mode):
    #=====================================
        """
        Generate Barabasi-Albert preferential attachment network. BA model starts with k initial nodes, and k edges 
        - each new node will connect to k nodes with p(number of edges already connected to each node). 
        
            edge_density = number of edges of each node
            mode = directed or undirected network
            
        """
        import numpy as np
        self.A = np.zeros(self.dist.shape) #initialise graph

        if mode == 'undirected':
            add_list = [[0,1], [1,0]]
            self.connect(edge_density, add_list)

        if mode == 'directed':
            add_list = [[0,1]]
            self.connect(edge_density, add_list)
            add_list = [[1,0]]
            self.connect(edge_density, add_list)
                
        return(self)
            
    
    #CALCULATE CYCLES
    #=================
    #=================
    #===========================
    def cycles_calculate(self, edge_density, mode):
    #===========================
        import networkx as nx
        import numpy as np
        
        cyc_mat = self.net_generate(edge_density, mode).A #matrix to calculate cycles
        G = nx.from_numpy_matrix(cyc_mat)
        cyc = nx.algorithms.cycle_basis(G)
        edge =  int(np.sum(cyc_mat))
        self.cycles = len(cyc)
        self.edges = edge
        return(self)
    
    
    #BUILD WEIGHT MATRIX
    #===================
    #===================

    # Conversion from distance to edge weights, scaled (itself exponentially) by s
    #====================================
    def dist2edge(self, distance, divisor, soften, s, r):
    #===================================
        import numpy as np
        self.edge_weight_out = (s + np.exp(-soften/np.exp(r)*distance))/divisor
        return(self)  

    #===========================
    def adjmat_generate(self, edge_density, s, r, divisor, soften, mode):
    #===========================
        import numpy as np
        import copy
        mat = np.zeros((self.dist.shape))
        
        curr_mat = self.net_generate(edge_density, mode).A #matrix to calculate cycles
        
        [rows, cols]    = np.where(curr_mat == 1) 
        
        
        #full_distance = np.linspace(0, np.max(self.dist), 300)
        #sumto = np.sum(self.dist2edge(full_distance, divisor, soften, s, 0).edge_weight_out)
        #curr_sum = np.sum(self.dist2edge(full_distance, divisor, soften, s, r).edge_weight_out)
        #factor = sumto/curr_sum
        
        for e in range(len(rows)):
            edge_weight = (self.dist2edge(self.dist[rows[e], cols[e]], divisor, soften, s, r).edge_weight_out)#* factor)
            mat[rows[e], cols[e]] = edge_weight 
                
        self.adj_mat = copy.deepcopy(mat)
            
        return(self)
    
    
    
    #SIMULATE AVALANCHES
    #===================
    #===================

    #Randomly select start node
    #===========================================
    def find_start_nodes(self, input_size, curr_mat):
    #===========================================
        import random
        import numpy as np
        start_nodes=set()
        for i in range(input_size):
            x = random.choice(np.arange(0,curr_mat.shape[0]))
            start_nodes.add(x)
            self.start_nodes = np.array(list(start_nodes))
        return(self)


    #Find cells to propagate
    #=====================================================
    def propagate_neighbours(self, curr_mat, start_node, thresh):
    #=====================================================
        import numpy as np
        self.prop_nodes = []
        nodes = np.where(curr_mat[start_node] > 0) [0]
        weights = curr_mat[start_node][nodes]
        for f in range(len(nodes)):
            if weights[f] > np.random.uniform(0, thresh):
                self.prop_nodes = np.append(self.prop_nodes, nodes[f])
        return(self)




    #Ping node
    #===========================
    def ping(self,  edge_density, r, s, divisor, soften, mode, n_sims, thresh, input_size, cutoff):
    #===========================
        import numpy as np
        curr_mat = self.adjmat_generate(edge_density, s, r,  divisor, soften, mode).adj_mat

        self.av_size = []
        self.av_dur = []

        #Simulate multiple times and take average for each input
        for i in range(n_sims):
            allstart_nodes = self.find_start_nodes(input_size, curr_mat).start_nodes
            t_nodes = [] #nodes at current time step activated

            #Find all nodes activate by pinged nodes
            for i in range(len(allstart_nodes)):
                #Initialise avalanche - ping first node
                start_node = allstart_nodes[i]
                t_nodes = np.append(t_nodes, self.propagate_neighbours(curr_mat, start_node, thresh).prop_nodes) #Find connected neighbours > threshold

            curr_list = t_nodes

            if len(t_nodes) > 1: #must have at least 3 cells to begin avalanche
                iterate = 'yes'
                all_nodes = t_nodes
                timesteps = 1

                while iterate == 'yes':
                    tplus_nodes = []
                    for z in range(len(curr_list)):
                        #List of all nodes active in next timestep
                        tplus_nodes = np.append(tplus_nodes, self.propagate_neighbours(curr_mat, int(curr_list[z]), thresh).prop_nodes)

                    all_nodes = np.append(all_nodes, tplus_nodes)
                    timesteps+=1
                    curr_list = tplus_nodes

                    if timesteps == cutoff:
                        iterate = 'no'

                    if len(tplus_nodes) == 0: #if no more active cells - stop
                        iterate = 'no'

                self.av_size = np.append(self.av_size, len(all_nodes)) 
                self.av_dur = np.append(self.av_dur, timesteps)
            else:
                continue
        
        
        
        return(self)
    

#==================================
def bin_data(spikes, N, sim_time):
#==================================
    """
    Bin spike data

    """
    import numpy as np
    bin_dat = np.zeros((N, sim_time))
    for i in range(N):
        bin_dat[i][np.unique((np.asarray(spikes[i])*1000).astype(int))] = 1
    return(bin_dat)



#===============================================================================
def run_net(sim_time, k, v_th, r, s, divisor, soften, N, dist, v_rest, t_syn_del, tau_l, N_e, lam, w_e):
#===============================================================================
    """
    Run spiking net using Brian2
    
    Inputs:
        sim_time (float): time steps to run simulation
        k (int): number of edges in network
        v_th (float): spike threshold 
        r (float): weight scaling parameter, defining local vs global scaling
        s (float): weight scaling parameter, defining overall range 
        divisor (float): divisor value for scaling function
        soften (float): degree of exponential softening for scaling function
        N (int): number of neurons in network
        dist (np array): distance matrix
        v_rest (float): resting membrane potential
        t_syn_del (float): synaptic delay
        tau_l (float): time constant
        N_e (int): number of external neurons
        lam (float): Poisson input rate
        w_e (float): weight from poisson inputs onto network
        
    Returns:
        bind (np.array): cells x timepoints, downsampled binarised array of spikes
        spikes (np array): cells x timepoints, full binarised array
        volt (np array): cells x timepoints, membrane potential
    
    """

    import brian2 as b2
    from random import sample
    from numpy import random
    import numpy as np
    
    b2.start_scope()
    
    # define dynamics for each cell
    lif ="""
    dv/dt = -(v-v_rest) / tau_l : 1 """
    net_dyn = b2.NeuronGroup(
    N, model=lif,
    threshold="v>v_th", reset="v = v_rest",
    method="euler")
    net_dyn.v = v_rest #set starting value for voltage

    p_input = b2.PoissonInput(net_dyn, "v", N_e,lam, w_e)
    
    #Network connectivity + weights
    curr = ba_netsim(dist).adjmat_generate(k, s, r, divisor, soften, 'directed')
    A = curr.A
    W = curr.adj_mat

    #Build synapses
    net_syn = b2.Synapses(net_dyn, net_dyn, 'w:1', on_pre="v+=w", delay=t_syn_del)
    rows, cols = np.nonzero(A)
    net_syn.connect(i = rows, j = cols)
    net_syn.w = W[rows, cols]

    spike_monitor = b2.SpikeMonitor(net_dyn)
    V = b2.StateMonitor(net_dyn, 'v', record=True)
    b2.run(sim_time*b2.ms)
    spikes = spike_monitor.spike_trains()
    volt = np.asarray(V.v)
    bind = bin_data(spikes, N, sim_time)
    
    return(bind, spikes, volt, spike_monitor )


#==============================
def MSE(empirical, model, k, alpha):
#==============================
    """
     #Find the MSE between 2 distributions in log space - alpha = 0.09
    """
    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt
    fig, axarr = plt.subplots(figsize = (5,3))

    binvec = np.append(empirical,model)
    mini = np.min(binvec)
    maxi = np.max(binvec)
    bins = 100000
    mod_hist = axarr.hist(model, bins=bins, range = (mini, maxi), density=True, histtype='step', linewidth = 1.5, cumulative=-1)
    mod_xaxis = np.log10(mod_hist[1])
    mod_yaxis = np.log10(mod_hist[0])
    emp_hist = axarr.hist(empirical, bins=bins, range = (mini, maxi), density=True, histtype='step', linewidth = 1.5, cumulative=-1)
    emp_xaxis = np.log10(emp_hist[1])
    emp_yaxis = np.log10(emp_hist[0])


    plt.close(fig)
    diff_sq = (emp_yaxis - mod_yaxis)**2

    if len(np.where(diff_sq == float('inf'))[0]) > 0:
        end_index = np.where(diff_sq == float('inf'))[0][0]
        diff_sq_full = diff_sq[:end_index]
        MSE = np.sum(diff_sq_full)/ (len(diff_sq_full) - k)

        res = emp_yaxis - mod_yaxis
        res_full = res[:end_index]
        var_res = np.sum((res_full - np.mean(res_full))**2)/len(res_full)

        empty_bins = bins - end_index
        Beta = (empty_bins *(10**-5))
        MSE_B = MSE + np.exp(Beta)*alpha

    else:
        MSE = np.sum(diff_sq)/ (len(diff_sq) - k)

        res = emp_yaxis - mod_yaxis
        var_res = np.sum((res - np.mean(res))**2)/len(res)

        empty_bins = 0
        Beta = (empty_bins *(10**-5))
        MSE_B = MSE + np.exp(Beta)*alpha
    return(MSE_B, MSE, var_res)

#==============================
def ks_log(empirical, model): 
#==============================
    """
    Find the distance between 2 distributions in log space

    """

    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt
    fig, axarr = plt.subplots(figsize = (5,3))
    binvec = np.append(empirical,model)
    mini = np.min(binvec)
    maxi = np.max(binvec)
    bins = 100000
    model_hist = axarr.hist(model, bins=bins, range = (mini, maxi), density=True, histtype='step', linewidth = 1.5, cumulative=-1)
    model_xaxis = np.log10(model_hist[1])
    model_yaxis = np.log10(model_hist[0])

    emp_hist = axarr.hist(empirical, bins=bins, range = (mini, maxi), density=True, histtype='step', linewidth = 1.5, cumulative=-1)
    emp_xaxis = np.log10(emp_hist[1])
    emp_yaxis = np.log10(emp_hist[0])

    mod_inf = np.where(model_yaxis == float('-inf'))[0]
    emp_inf = np.where(emp_yaxis == float('-inf'))[0]
    plt.close(fig)
        
    if len(emp_inf) == 0 and len(mod_inf) == 0:
        end_index = len(emp_inf)

    elif len(emp_inf) == 0:
        end_index = mod_inf[0] 

    elif len(mod_inf) == 0:
        end_index = emp_inf[0] 
        

    diff_vec = abs(abs(model_yaxis[:end_index]) - abs(emp_yaxis[:end_index ]))

    cost_max, cost_mean = np.max(diff_vec), np.mean(diff_vec)

    return(cost_max, cost_mean)


#==============================
def num_sims(empirical, cutoff):
#==============================
    """
    Calculate number of simulatons to do - to have 95% chance of generating maximum avalanche
    """
    import matplotlib.pyplot as plt
    import math
    fig, axarr = plt.subplots(figsize = (7,5))
    hist = axarr.hist(empirical, bins = 100000, density = True, histtype = 'step', cumulative = -1)
    p = 1-(10**(np.log10(hist[0])[np.where(np.log10(hist[1]) > np.log10(cutoff))[0][0]])) #probability of getting avalanches of size cutoff or smaller
    number = 0.05 
    base = p
    exponent = int(math.log(number, base)) #number of simulations as the power to which p is raised to get 95% probability 
    return(exponent)



#==========================================
def sub_sweep(data, const_list, val_list):
#==========================================
    """
    Sweep through parameter combinations while keeping certain parameter combinations constant

    """

    import numpy as np
    if len(const_list) == 1:
        par = const_list[0]
        val = val_list[0]

        if par == 'k':
            index = 0
        if par == 'v_th':
            index = 1
        if par == 'r':
            index = 2

        where = []
        for i in range(len(data)):
            if str(data[i][0][index]) == str(val):
                where = np.append(where, i)
        where = where.astype(int)


        output_list = list(range(len(where)))
        for i in range(len(where)):
            output_list[i] = data[where[i]]

    if len(const_list) > 1:

        where_list = list(range(len(const_list)))
        for x in range(len(const_list)):
            par = const_list[x]
            val = val_list[x]

            if par == 'k':
                index = 0
            if par == 'v_th':
                index = 1
            if par == 'r':
                index = 2

            where = []
            for i in range(len(data)):
                if str(data[i][0][index]) == str(val):
                    where = np.append(where, i)
            where = where.astype(int)
            where_list[x] = where
        inter = np.intersect1d(where_list[0], where_list[1])

        output_list = list(range(len(inter)))
        for i in range(len(inter)):
            output_list[i] = data[inter[i]]
        
    return(output_list)
  
