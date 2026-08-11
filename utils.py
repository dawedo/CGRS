import numpy as np

# --------------------------
# Build virtual profile weights
# --------------------------
# def Virtual_user_weights(profiles_train, group,group_items, G,support, coCount, nusers):
#     weights = np.zeros(len(group_items))
#     for idx, item in enumerate(group_items):
#         item_score = 0.0
#         for user in group:
#             if (item in profiles_train[user]): #or (item in V[user]):
#                 item_score += 1
#             else:
#                 item_score += scoreP(profiles_train[user], item, support, coCount, nusers)
#         weights[idx] = item_score / G
#     return weights
# Remove the function entirely or simplify to:
def Personal_user_weights(profiles_train, user, user_items, support, coCount, nusers):
    weights = np.zeros(len(user_items))
    for idx, item in enumerate(user_items):
        if item in profiles_train[user]:
            weights[idx] = 1.0
        else:
            weights[idx] = scoreP(profiles_train[user], item, support, coCount, nusers)
    return weights
def coexistenceCount(profiles, ITEMS):
    coCount  = np.zeros((ITEMS, ITEMS), dtype=int)
    #print(coexistenceCount.shape)
    for profile in profiles:
        prof = np.array(profile, dtype=int)
        # Build mask for all co-occurrences in this profile
        coCount[np.ix_(prof, prof)] += 1
        
    np.fill_diagonal(coCount, 0)  # zero out diagonal at the end
    return coCount

def scoreP(profile, item, support, coCount, nusers):
    # Vectorized computation
    pp = coCount[profile, item].astype(float) / support[profile].astype(float)

    #pp = coCount[profile, item] / support[profile]
    """print(f" profile={profile}, item={item}, type(profile)={type(profile)}, type(item)={type(item)}")
    try:
        pp = coCount[int(profile), int(item)] / support[int(profile)]
    except Exception as e:
        print(f"Error at iteration: profile={profile}, item={item}, type(profile)={type(profile)}, type(item)={type(item)}")
        raise e"""
    
    # Get top-2 values without full sort
    if len(pp) >= 2:
        top2 = np.partition(pp, -2)[-2:]
    else:
        top2 = pp
    #top2 = np.partition(pp, -2)[-2:]  # last 2 largest
    sc = support[item] / nusers
    sc *= np.prod(top2)  # multiply both
    return sc

def scoreR(profile, item, profile_weights, support, coCount, nusers):
    # profile_weights is a vector (like profile_wt in your earlier code)
    #print(coexistenceCount.shape, profile_weights.shape)
    #pp = profile_weights * coCount[profile, item] / support[profile]
    
    pp = profile_weights * (coCount[profile, item].astype(float) / support[profile].astype(float))

    #top2 = np.partition(pp, -2)[-2:]
    if len(pp) >= 2:
        top2 = np.sort(pp)[-2:]
    else:
        top2 = pp
    sc = support[item] / nusers
    sc *= np.prod(top2)
    return sc

def findLessCount(PP, pp_nv):
    # assumes PP is sorted ascending
    return np.searchsorted(PP, pp_nv, side='right')

# def modify(group, val, profiles_train, profiles_test, support, coCount):
#     for user in group:
#         test_items = np.array(profiles_test[user], dtype=int)
#         train_items = np.array(profiles_train[user], dtype=int)

#         # update support counts
#         support[test_items] += val

#         # update coexistence counts in one shot
#         coCount[np.ix_(train_items, test_items)] += val
#         coCount[np.ix_(test_items, train_items)] += val

#     return support, coCount


def modify(user, val, profiles_train, profiles_test, support, coCount):
    test_items = np.array(profiles_test[user], dtype=int)
    train_items = np.array(profiles_train[user], dtype=int)

    # update support counts
    support[test_items] += val

    # update coexistence counts in one shot
    coCount[np.ix_(train_items, test_items)] += val
    coCount[np.ix_(test_items, train_items)] += val

    return support, coCount