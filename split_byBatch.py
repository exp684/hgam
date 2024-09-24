from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True) # Set "nopython" mode for best performance.
#@jit(nopython=True, parallel=True, cache=True, fastmath=True)  # Set "nopython" mode for best performance.
def extractRoute(route, myRoute): 
    # Extracting the route. Positions 0 and 1 are the depot.        
    batch_size = len(route)
    nb_nodes = len(route[0])
    for batch in prange(batch_size): 
        pos = 0
        for i in range(nb_nodes):
            if ( route[batch, i] != 0 and route[batch, i] != 1 ):
                myRoute[batch, pos] = route[batch, i]
                pos +=1        

    return myRoute 

@jit(nopython=True, parallel=True) # Set "nopython" mode for best performance.
#@jit(nopython=True, parallel=True, cache=True, fastmath=True)  # Set "nopython" mode for best performance.
def distanceFrom(locations, myRoute, myDistance, position): 
    # To compute the distance between a given 'position' (i.e. the depot at the begining or the end) and all other locations/clients.
    batch_size = len(myRoute)
    nb_nodes = len(myRoute[0])
    for batch in prange(batch_size):
        for i in range(nb_nodes):
            current = myRoute[batch, i]
            myDistance[batch, i] = np.sqrt( np.sum( np.square(locations[batch, current] - locations[batch, position]) ) )

    return myDistance

@jit(nopython=True, parallel=True) # Set "nopython" mode for best performance.
#@jit(nopython=True, parallel=True, cache=True, fastmath=True)  # Set "nopython" mode for best performance.
def distanceBetween(locations, myRoute, myDistance): 
    # To compute the distance between to consequtive positions/clients.
    batch_size = len(myRoute)
    nb_nodes = len(myRoute[0])
    for batch in prange(batch_size):
        for i in range(nb_nodes - 1):
            current = myRoute[batch, i]
            next = myRoute[batch, i + 1]
            myDistance[batch, i] = np.sqrt( np.sum( np.square(locations[batch, current] - locations[batch, next]) ) )

    return myDistance

@jit(nopython=True, parallel=True) # Set "nopython" mode for best performance.
#@jit(nopython=True, parallel=True, cache=True, fastmath=True)  # Set "nopython" mode for best performance.
def buildVectors(scores, Tmax, myRoute, distance_from_start, distance_to_finish, distance_to_next, vp, vc, vs): 
    # build vector of size (vl), profit (vp), length/cost (vc) and successor (vs) for each saturated tour.    
    batch_size = len(myRoute)
    nb_nodes = len(myRoute[0])
    for batch in prange(batch_size):
        profit, length, lengthest = 0, 0, 0
        for u in range(nb_nodes):
            profit = scores[batch, myRoute[batch, u]]
            length = distance_from_start[batch, u]
            #for v in range(u+1, nb_nodes):
            v = u + 1
            while(v < nb_nodes):
                lengthest = length + distance_to_next[batch, v-1]
                if(lengthest + distance_to_finish[batch, v] > Tmax[batch]):
                    break
                else:
                    length = lengthest
                    profit = profit + scores[batch, myRoute[batch, v]]
                v+=1
            length = length + distance_to_finish[batch, v-1]
            vs[batch, u] = v  
            if length <= Tmax[batch] : vp[batch, u] = profit
            else: vp[batch, u] = 0
            vc[batch, u] = length

    return vp, vc, vs

@jit(nopython=True, parallel=True) # Set "nopython" mode for best performance.
#@jit(nopython=True, parallel=True, cache=True, fastmath=True)  # Set "nopython" mode for best performance.
def takeOptimal(m, vp, vs, mprofit, sc): 
    # 2 - dynamic programming to find the maximum-weighted independent set.
    batch_size = len(vp)   
    nb_nodes = len(vp[0])

    for batch in prange(batch_size):
        oldprofit, newprofit = 0, 0 
        # last line (make sure to call the function with n>0).
        mprofit[batch, nb_nodes-1, 0] = vp[batch, nb_nodes-1]
        for u in prange(1, m[batch]): mprofit[batch, nb_nodes-1, u] = 0

        for u in range(nb_nodes-1):
            for v in range(m[batch]):
                oldprofit = mprofit[batch, nb_nodes-2-u+1, v]
                newprofit = vp[batch, nb_nodes-2-u]
                if v>0 and vs[batch, nb_nodes-2-u] < nb_nodes:
                    newprofit = newprofit + mprofit[batch, vs[batch, nb_nodes-2-u], v-1]
                if newprofit>oldprofit:
                    mprofit[batch, nb_nodes-2-u, v] = newprofit 
                else:
                    mprofit[batch, nb_nodes-2-u, v] = oldprofit 
        # 3 - back track to find the saturated tours
        maxj = 0

        for v in prange(1, m[batch]):
            if mprofit[batch, 0, v] > mprofit[batch, 0, maxj]: maxj = v

        # Taking the optimal solution of the problem
        sc[batch] = mprofit[batch, 0, maxj]
    
    return sc

@jit(nopython=True, parallel=True) # Set "nopython" mode for best performance.
#@jit(nopython=True, parallel=True, cache=True, fastmath=True)  # Set "nopython" mode for best performance.
def generateNodes(size, Tmax, nodes):        
    dx = np.random.rand()  # Coordinate x start/end depot. 
    dy = np.random.rand()  # Coordinate y start/end depot.
    nodes[0, 0], nodes[1, 0] = dx, dx
    nodes[0, 1], nodes[1, 1] = dy, dy
    for i in prange(2, size):
        reach = False
        while(reach==False):
            x = np.random.rand()    # Coordinate x node. 
            y = np.random.rand()    # Coordinate y node.
            dn = np.sqrt(  np.square(dx - x) + np.square(dy - y)  )                    
            if(2*dn <= Tmax): reach = True 
        nodes[i, 0] = x
        nodes[i, 1] = y

    return nodes