
import numpy as np
from operator import itemgetter

def conformal(cand_items, group_train, V_train, train_weights, coCount, n):
    
    

    """p1 = {}
    for item in cand_items:
        p1[item] = 0 
    for v in V_train:
        coexist = coCount[group_train, v]
        wt_sup = train_weights * coexist

        for item in cand_items:
            p1[item] += ((wt_sup >= coCount[item, v]).sum()+1)/(n+1)
    for item in cand_items:
        p1[item] /= k"""

    k = len(V_train)
    p = np.zeros(len(cand_items), dtype=float)

    for v in V_train:
        coexist = coCount[group_train, v]     # shape: (len(group_train),)
        wt_sup = train_weights * coexist  #coexist     # shape: (len(group_train),)
        # Compare wt_sup with coCount[item, v] for all candidate items
        co_v = coCount[cand_items, v]              # shape: (len(cand_items),)
        # Broadcasting comparison:    # (len(cand_items), len(group_train))
        comp = wt_sup <= co_v[:, None]              # Compare each candidate vs group
        p += (comp.sum(axis=1)+1) / (n+1)                  # Sum over group dimension

    p /= k
    
    sorted_indices = np.argsort(-p)
    sorted_cand_items = cand_items[sorted_indices]
    #recommendations_topK = sorted_cand_items[:top_K]
    
    return sorted_cand_items
