
import numpy as np

def evaluate_personal_ranking_metrics(user, profiles_test, cand_items, K_values):
    test_items = set(profiles_test[user])
    n_relevant = len(test_items)
    
    if n_relevant == 0:
        return {}
    
    relevance = np.isin(cand_items, list(test_items)).astype(int)
    metrics = {}

    # ---------- 1. Cutoff-based metrics ----------
    for K in K_values:
        #rel_k = relevance[:K]
        rel_k = np.isin(cand_items[:K], list(test_items)).astype(int)
        tp = rel_k.sum()

        precision = tp / K #min(K, n_relevant)
        recall = tp / n_relevant 
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        metrics[f"precision@{K}"] = precision
        metrics[f"recall@{K}"] = recall
        metrics[f"f1@{K}"] = f1

    # ---------- 2. Average Precision ----------
    precisions = np.cumsum(relevance) / (np.arange(len(relevance)) + 1)
    metrics["ap"] = (np.sum(precisions * relevance) / n_relevant if n_relevant > 0 else 0 )

    # ---------- 3. Reciprocal Rank ----------
    hit_indices = np.where(relevance == 1)[0]
    metrics["rr"] = 1.0 / (hit_indices[0] + 1) if len(hit_indices) > 0 else 0

    # ---------- 4. nDCG ----------
    gains = (2**relevance - 1) / np.log2(np.arange(2, len(relevance) + 2))
    DCG = gains.sum()
    ideal_relevance = np.sort(relevance)[::-1]
    IDCG = ((2**ideal_relevance - 1) / np.log2(np.arange(2, len(ideal_relevance) + 2))).sum()
    metrics["nDCG"] = DCG / IDCG if IDCG > 0 else 0

    # ---------- 5. AUC ----------
    pos = np.where(relevance == 1)[0]
    neg = np.where(relevance == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        auc = 0.0
    else:
        total_pairs = len(pos) * len(neg)
        auc = np.mean([i < j for i in pos for j in neg])
    metrics["AUC"] = auc

    

    return metrics
