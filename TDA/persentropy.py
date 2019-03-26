def persentropy(dgms):
    """
    Calculates the persistent entropies of a set of persistent diagrams.
    Author: Daniele Marchei
    """
    import numpy as np
    import math
    result = []

    #calculate PE for all diagrams
    for dgm in dgms:
        dgm_np = np.array(dgm)
        
        #remove the point at infinity
        inf_index = [i for i,k in enumerate(dgm_np) if k[1] == math.inf]
        if len(inf_index) > 0:
            dgm_np = np.delete(dgm_np, inf_index,0)

        L = np.sum(dgm_np[0:,1] - dgm_np[0:,0])
        ls = dgm_np[0:,1] - dgm_np[0:,0]
        ps = ls / L
        Hs = ps * np.log(ps)
        H = - np.sum(Hs)
        result.append(H)
        
    return np.array(result)

