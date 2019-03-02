from __future__ import division
from math import log,sqrt
import operator
from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
    return o_counts[word2wid[word]]

def tw_stemmer(word):
    '''Stems the word using Porter stemmer, unless it is a
    username (starts with @).  If so, returns the word unchanged.

    :type word: str
    :param word: the word to be stemmed
    :rtype: str
    :return: the stemmed word

    '''
    if word[0] == '@': #don't stem these
        return word
    else:
        return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N):
    '''Compute the pointwise mutual information using cooccurrence counts.

    :type c_xy: int
    :type c_x: int
    :type c_y: int
    :type N: int
    :param c_xy: coocurrence count of x and y
    :param c_x: occurrence count of x
    :param c_y: occurrence count of y
    :param N: total observation count
    :rtype: float
    :return: the pmi value

    '''
    p = (c_xy / N) / ((c_x / N) * (c_y / N))
    pmi = log(p, 2)
  
    return pmi

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

def PPMI_alpha(c_xy, c_x, c_y, N, alpha=0.75):
    '''Compute the pointwise mutual information using cooccurrence counts.

    :type c_xy: int
    :type c_x: int
    :type c_y: int
    :type N: int
    :type alpha: float
    :param c_xy: coocurrence count of x and y
    :param c_x: occurrence count of x
    :param c_y: occurrence count of y
    :param N: total observation count
    :param alpha: raises contexts to the power of alpha
    :rtype: float
    :return: the pmi_alpha value

    '''
    p = (c_xy / N) / ((c_x / N) * ((c_y / N)**alpha))
    ppmi_alpha = log(p, 2)
  
    return ppmi_alpha

#Do a simple error check using value computed by hand
if(PPMI_alpha(2,4,1,16) != 2): # these numbers are from our y,z example
    print("Warning: PPMI_alpha is incorrectly defined")
else:
    print("PPMI_alpha check passed")

def lap_smooth(c_co, id_dict=wid2word, k=2.):
    '''Compute the pointwise mutual information using cooccurrence counts.

    :type c_co: dict
    :type k: float
    :param c_co: original co_counts
    :param k: constant k for laplace smoothing
    :rtype: dict
    :return: c_co_lap

    '''
    c_co_lap = {}
    for id0_t in c_co.keys():
        d_t = c_co[id0_t]
        for id1_t in id_dict.keys():
            if id1_t not in d_t.keys():
                d_t[id1_t] = k
            else:
                d_t[id1_t] += k
        c_co_lap[id0_t] = d_t
    return c_co_lap

def ttest(c_xy, c_x, c_y, N):
    '''Compute by t-test using cooccurrence counts.

    :type c_xy: int
    :type c_x: int
    :type c_y: int
    :type N: int
    :param c_xy: coocurrence count of x and y
    :param c_x: occurrence count of x
    :param c_y: occurrence count of y
    :param N: total observation count
    :rtype: float
    :return: the result computed by t-test

    '''
    p_xy = c_xy / N
    p_x = c_x / N
    p_y = c_y / N
    
    ttest_result = (p_xy - p_x * p_y) / sqrt(p_x * p_y)
  
    return ttest_result

#Do a simple error check using value computed by hand
if(ttest(1,1,1,2) != 0.5): # these numbers are from our y,z example
    print("Warning: t-test is incorrectly defined")
else:
    print("t-test check passed")

def minkow_dist(v0,v1,p=2):
    '''Compute the Minkowski distance between two sparse vectors.

    :type v0: dict
    :type v1: dict
    :type p: int
    :param v0: first sparse vector
    :param v1: second sparse vector
    :param p:
    :rtype: float
    :return: Minkowski distance between v0 and v1
    '''
    v0_keys = v0.keys()
    v1_keys = v1.keys()

    v01_k = []
    v0_k = []
    v1_k = []

    temp = 0

    # find the keys in both dicts and perticular dict
    for key0 in v0_keys:
        if key0 in v1_keys:
            v01_k.append(key0)
        else:
            v0_k.append(key0)

    for key1 in v1_keys:
        if key1 not in v0_keys:
            v1_k.append(key1)

    # compute
    for i in v01_k:
        temp += (v0[i] - v1[i])**p

    for j in v0_k:
        temp += v0[j]**p

    for k in v1_k:
        temp += (-v1[k])**p

    minkov_dist = -temp**(1/p)

    return minkov_dist

def cos_sim(v0,v1):
    '''Compute the cosine similarity between two sparse vectors.

    :type v0: dict
    :type v1: dict
    :param v0: first sparse vector
    :param v1: second sparse vector
    :rtype: float
    :return: cosine between v0 and v1
    '''
    # We recommend that you store the sparse vectors as dictionaries
    # with keys giving the indices of the non-zero entries, and values
    # giving the values at those dimensions.

    #You will need to replace with the real function

    # calculate the numerator
    v0_keys = v0.keys()
    v1_keys = v1.keys()

    nu = 0

    for key in v0_keys:
      if key in v1_keys:
          nu += v0[key] * v1[key]

    # calculate the denominator
    de_0 = 0
    de_1 = 0
    for i in v0.values():
        de_0 += i ** 2
    for j in v1.values():
        de_1 += j ** 2
    de = sqrt(de_0) * sqrt(de_1)

    cos = nu / de

    return cos

def jac_sim(v0,v1):
    '''Compute the Jaccard similarity between two sparse vectors.

    :type v0: dict
    :type v1: dict
    :param v0: first sparse vector
    :param v1: second sparse vector
    :rtype: float
    :return: Jaccard index between v0 and v1
    '''
    # make v0 and v1 with same dimension
    key_v0 = v0.keys()
    key_v1 = v1.keys()
    test_v0 = {}
    test_v1 = {}
    for key_temp0 in key_v0:
        test_v0[key_temp0] = v0[key_temp0]
        if key_temp0 not in key_v1:
            test_v1[key_temp0] = 0
    for key_temp1 in key_v1:
        test_v1[key_temp1] = v1[key_temp1]
        if key_temp1 not in key_v0:
            test_v0[key_temp1] = 0
    
    # compute the Jaccard similarity    
    min_sum = 0.
    max_sum = 0.
    for t in test_v0:
        min_sum += min(test_v0[t], test_v1[t])
        max_sum += max(test_v0[t], test_v1[t])
    
    return min_sum / max_sum

# test Jaccard
v0_t = {1:2, 2:4}
v1_t = {0:1, 3:5, 2:6}
if jac_sim(v0_t,v1_t) != 4/14:
    print('Warning: Jaccard check fail')
else:
    print('Jaccard check passed')

def create_pmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors_pmi = {}
    for wid0 in wids:
        ##you will need to change this
        c_wid0 = o_counts[wid0]
        co_dict = co_counts[wid0]
        v_temp = {}
        for wid1 in co_dict.keys():
            if wid1 != wid0:
                c_wid1 = o_counts[wid1]
                c_wid01 = co_dict[wid1]
                pmi = PMI(c_wid01, c_wid0, c_wid1, tot_count)
                v_temp[wid1] = pmi
        vectors_pmi[wid0] = v_temp

    for k_temp in vectors_pmi.keys():
        if vectors_pmi[k_temp] == 0:
            vectors_pmi.pop(k_temp)

    return vectors_pmi

def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors_ppmi = {}
    for wid0 in wids:
        c_wid0 = o_counts[wid0]
        co_dict = co_counts[wid0]
        v_temp = {}
        for wid1 in co_dict.keys():
            if wid1 != wid0:
                c_wid1 = o_counts[wid1]
                c_wid01 = co_dict[wid1]
                pmi = PMI(c_wid01, c_wid0, c_wid1, tot_count)
                if pmi < 0:
                    pmi = 0
                v_temp[wid1] = pmi
        vectors_ppmi[wid0] = v_temp

    for k_temp in vectors_ppmi.keys():
        if vectors_ppmi[k_temp] == 0:
            vectors_ppmi.pop(k_temp)

    return vectors_ppmi

def create_ppmi_alpha_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PMI_alpha.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors_alpha = {}
    for wid0 in wids:
        c_wid0 = o_counts[wid0]
        co_dict = co_counts[wid0]
        v_temp = {}
        for wid1 in co_dict.keys():
            if wid1 != wid0:
                c_wid1 = o_counts[wid1]
                c_wid01 = co_dict[wid1]
                pmi = PPMI_alpha(c_wid01, c_wid0, c_wid1, tot_count)
                v_temp[wid1] = max(0, pmi)
        vectors_alpha[wid0] = v_temp

# =============================================================================
#     for k_temp in vectors_alpha.keys():
#         if vectors_alpha[k_temp] == 0:
#             vectors_alpha.pop(k_temp)
# =============================================================================

    return vectors_alpha

def create_ppmi_lap_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PPMI_lap.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    c_co_lap = lap_smooth(co_counts, k=2.)
    vectors_ppmi_lap = {}
    for wid0 in wids:
        c_wid0 = o_counts[wid0]
        co_dict = c_co_lap[wid0]
        v_temp = {}
        for wid1 in co_dict.keys():
            if wid1 != wid0:
                c_wid1 = o_counts[wid1]
                c_wid01 = co_dict[wid1]
                pmi = PMI(c_wid01, c_wid0, c_wid1, tot_count)
                if pmi < 0:
                    pmi = 0
                v_temp[wid1] = pmi
        vectors_ppmi_lap[wid0] = v_temp

# =============================================================================
#     for k_temp in vectors_ppmi_lap.keys():
#         if vectors_ppmi_lap[k_temp] == 0:
#             vectors_ppmi_lap.pop(k_temp)
# =============================================================================

    return vectors_ppmi_lap

def create_ttest_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using t_test.
    These should be sparse vectors.

    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors_ttest = {}
    for wid0 in wids:
        c_wid0 = o_counts[wid0]
        co_dict = co_counts[wid0]
        v_temp = {}
        for wid1 in co_dict.keys():
            if wid1 != wid0:
                c_wid1 = o_counts[wid1]
                c_wid01 = co_dict[wid1]
                ttest_result = ttest(c_wid01, c_wid0, c_wid1, tot_count)
                v_temp[wid1] = ttest_result
        vectors_ttest[wid0] = v_temp

# =============================================================================
#     for k_temp in vectors_ttest.keys():
#         if vectors_ttest[k_temp] == 0:
#             vectors_ttest.pop(k_temp)
# 
# =============================================================================
    return vectors_ttest

def read_counts(filename, wids):
    '''Reads the counts from file. It returns counts for all words, but to
    save memory it only returns cooccurrence counts for the words
    whose ids are listed in wids.

    :type filename: string
    :type wids: list
    :param filename: where to read info from
    :param wids: a list of word ids
    :returns: occurence counts, cooccurence counts, and tot number of observations
    '''
    o_counts = {} # Occurence counts
    co_counts = {} # Cooccurence counts
    fp = open(filename)
    N = float(next(fp))
    for line in fp:
        line = line.strip().split("\t")
        wid0 = int(line[0])
        o_counts[wid0] = int(line[1])
        if(wid0 in wids):
                co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
    return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, name, first=0, last=-1):
    '''Sorts the pairs of words by their similarity scores and prints
    out the sorted list from index first to last, along with the
    counts of each word in each pair.

    :type similarities: dict
    :type o_counts: dict
    :type first: int
    :type last: int
    :param similarities: the word id pairs (keys) with similarity scores (values)
    :param o_counts: the counts of each word id
    :param first: index to start printing from
    :param last: index to stop printing
    :return: none
    '''
    file = './sim_result/' + name + '.txt'
    f = open(file, 'w')
    if first < 0: last = len(similarities)
    for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
        word_pair = (wid2word[pair[0]], wid2word[pair[1]])
        print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair), o_counts[pair[0]],o_counts[pair[1]]))
        f.write("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair), o_counts[pair[0]],o_counts[pair[1]]))
        f.write('\n')
    
    f.close()

def freq_v_sim(sims, name):
    xs = []
    ys = []
    for pair in sims.items():
        ys.append(pair[1])
        c0 = o_counts[pair[0][0]]
        c1 = o_counts[pair[0][1]]
        xs.append(min(c0,c1))
    plt.clf() # clear previous plots (if any)
    plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
    plt.plot(xs, ys, 'k.') # create the scatter plot
    plt.xlabel('Min Freq')
    plt.ylabel('Similarity')
    print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
    plt.savefig('./plot/' + name + '.pdf')
    plt.show() #display the set of plots
    
    file = './sim_result/' + name + '.txt'
    f = open(file, 'a')
    f.write("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
    f.write('\n')
    f.close()

def make_pairs(items):
    '''Takes a list of items and creates a list of the unique pairs
    with each pair sorted, so that if (a, b) is a pair, (b, a) is not
    also included. Self-pairs (a, a) are also not included.

    :type items: list
    :param items: the list to pair up
    :return: list of pairs

    '''
    return [(x, y) for x in items for y in items if x < y]

# the data for preliminary part
#test_words = ["cat", "dog", "mouse", "computer","@justinbieber"]
# other test data
#test_words = ["red", "yellow", "blue", "green", "lime", "cyan", "turquoise",
#              "azure", "indigo", "violet", "purple", "pink", "orange", "magenta",
#              "white", "grey", "black", "brown", "gold", "silver"]
#test_words = ["hate", "love", "@justinbieber", "#nike"]

# list of theme words from Contextual Correlates of Synonymy, H. Rubenstein, J. B. Goodenough, 1965
test_words = ['asylum', 'autograph', 'boy', 'brother', 'car', 'coast', 'cock', 'cord', 'crane', 'cushion', 'food', 'furnace', 'gem', 'glass', 'graveyard', 'grin', 'mound', 'noon', 'oracle', 'slave', 'tool', 'voyage', 'wizard', 'woodland']
#test_words = ['asylum', 'autograph', 'boy', 'brother', 'car', 'coast', 'cock', 'cord', 'crane', 'cushion', 'food', 'furnace', 'gem', 'glass', 'graveyard', 'grin', 'mound', 'noon', 'oracle', 'slave', 'tool', 'voyage', 'wizard', 'woodland', 'automobile', 'bird', 'cemetery', 'forest', 'fruit', 'hill', 'implement', 'jewel', 'journey', 'lad', 'madhouse', 'magician', 'midday', 'monk', 'pillow', 'rooster', 'sage', 'shore', 'signature', 'smile', 'stove', 'string', 'tumbler']

stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs
wid_pairs = make_pairs(all_wids)

#read in the count information
(o_counts, co_counts, N) = read_counts("/afs/inf.ed.ac.uk/group/teaching/anlp/asgn3/counts", all_wids)

# create vectors
vectors_pmi = create_pmi_vectors(all_wids, o_counts, co_counts, N)
vectors_ppmi = create_ppmi_vectors(all_wids, o_counts, co_counts, N)
vectors_ppmi_alpha = create_ppmi_alpha_vectors(all_wids, o_counts, co_counts, N)
vectors_ppmi_lap = create_ppmi_lap_vectors(all_wids, o_counts, co_counts, N)
vectors_ttest = create_ttest_vectors(all_wids, o_counts, co_counts, N)

# =============================================================================
# # PMI + cos-----
# name = 'PMI_cos'
# pmi_c = {(wid0,wid1): cos_sim(vectors_pmi[wid0],vectors_pmi[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by cosine similarity with PMI")
# print_sorted_pairs(pmi_c, o_counts, name)
# freq_v_sim(pmi_c, name)
# print()
# =============================================================================

# =============================================================================
# # PPMI + cos-----
# name = 'PPMI_cos'
# ppmi_c = {(wid0,wid1): cos_sim(vectors_ppmi[wid0],vectors_ppmi[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by cosine similarity with PPMI")
# print_sorted_pairs(ppmi_c, o_counts, name)
# freq_v_sim(ppmi_c, name)
# print()
# =============================================================================

# =============================================================================
# # PPMI + Minkowski distance(p=2)-----
# name = 'PPMI_Min_dist_p_2'
# ppmi_m_2 = {(wid0,wid1): minkow_dist(vectors_ppmi[wid0],vectors_ppmi[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by Minkowski distance(p=2) with PPMI")
# print_sorted_pairs(ppmi_m_2, o_counts, name)
# freq_v_sim(ppmi_m_2, name)
# print()
# =============================================================================

# =============================================================================
# # PPMI + Jaccard index-----
# name = 'PPMI_Jaccard'
# ppmi_j = {(wid0,wid1): jac_sim(vectors_ppmi[wid0],vectors_ppmi[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by Jaccard index with PPMI")
# print_sorted_pairs(ppmi_j, o_counts, name)
# freq_v_sim(ppmi_j, name)
# print()
# =============================================================================

# =============================================================================
# # PPMI_alpha + cos-----
# name = 'PPMI_alpha_cos'
# ppmi_a_c = {(wid0,wid1): cos_sim(vectors_ppmi_alpha[wid0],vectors_ppmi_alpha[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by cosine similarity with PPMI_alpha")
# print_sorted_pairs(ppmi_a_c, o_counts, name)
# freq_v_sim(ppmi_a_c, name)
# print()
# =============================================================================

# =============================================================================
# # PPMI_alpha + Minkowski distance(p=2)-----
# name = 'PPMI_alpha_Min_dist_p_2'
# ppmi_a_m_2 = {(wid0,wid1): minkow_dist(vectors_ppmi_alpha[wid0],vectors_ppmi_alpha[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by Minkowski distance(p=2) with PPMI_alpha")
# print_sorted_pairs(ppmi_a_m_2, o_counts, name)
# freq_v_sim(ppmi_a_m_2, name)
# print()
# =============================================================================

# =============================================================================
# # PPMI_alpha + Jaccard index-----
# name = 'PPMI_alpha_Jaccard'
# ppmi_a_j = {(wid0,wid1): jac_sim(vectors_ppmi_alpha[wid0],vectors_ppmi_alpha[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by Jaccard index with PPMI_alpha")
# print_sorted_pairs(ppmi_a_j, o_counts, name)
# freq_v_sim(ppmi_a_j, name)
# print()
# =============================================================================

# =============================================================================
# # PPMI_laplace_smoothed + cos-----
# name = 'PPMI_lap_cos'
# ppmi_ls_c = {(wid0,wid1): cos_sim(vectors_ppmi_lap[wid0],vectors_ppmi_lap[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by cosine similarity with PPMI_laplace_smoothed")
# print_sorted_pairs(ppmi_ls_c, o_counts, name)
# freq_v_sim(ppmi_ls_c, name)
# print()
# 
# # PPMI_laplace_smoothed + Minkowski distance(p=2)-----
# name = 'PPMI_lap_Min_dist_p_2'
# ppmi_ls_m_2 = {(wid0,wid1): minkow_dist(vectors_ppmi_lap[wid0],vectors_ppmi_lap[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by Minkowski distance(p=2) with PPMI_laplace_smoothed")
# print_sorted_pairs(ppmi_ls_m_2, o_counts, name)
# freq_v_sim(ppmi_ls_m_2, name)
# print()
# 
# # PPMI_laplace_smoothed + Jaccard index-----
# name = 'PPMI_lap_Jaccard'
# ppmi_ls_j = {(wid0,wid1): jac_sim(vectors_ppmi_lap[wid0],vectors_ppmi_lap[wid1]) for (wid0,wid1) in wid_pairs}
# print("Sort by Jaccard index with PPMI_laplace_smoothed")
# print_sorted_pairs(ppmi_ls_j, o_counts, name)
# freq_v_sim(ppmi_ls_j, name)
# print()
# =============================================================================

# t-test + Jaccard index-----
name = 'ttest_Jaccard'
ttest_j = {(wid0,wid1): jac_sim(vectors_ttest[wid0],vectors_ttest[wid1]) for (wid0,wid1) in wid_pairs}
print("Sort by Jaccard index with t-test")
print_sorted_pairs(ttest_j, o_counts, name)
freq_v_sim(ttest_j, name)
print()
