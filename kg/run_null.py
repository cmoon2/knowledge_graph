'''
Created on Sep 3, 2016

@author: Changsung Moon (cmoon2@ncsu.edu)
'''


#!/usr/bin/env python


#import sys
#sys.path.append("[PATH OF THE FOLDER OF SOURCE CODES]")
#print(sys.path)

import numpy as np
from numpy.random import randint
from random import sample
from random import shuffle


from func.io import extract_ent_rel_id
from func.io import extract_data_conv_id
from func.io import extract_ent_rel_in_out
from func.io import extract_rel_in_out_ent

from func.util import extract_union_rel




def extract_ent(p, rel_ent):
    list_ent = []

    if rel_ent.has_key(p):
        list_ent = list(set(rel_ent[p].keys()))

    return list_ent


''' Freebase or YAGO '''
#data_set = "Freebase"
data_set = "YAGO"


''' Hits@N '''
hitsAtN = 10
results_rel = np.zeros(hitsAtN)
results_ent = np.zeros(hitsAtN)


''' Datasets '''
if data_set == "Freebase":
    kg_train = "../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-train.txt"
    kg_test = "../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-test.txt"
elif data_set == "YAGO":
    kg_train = "../../../../Datasets/YAGO/YAGO43k/YAGO43k_name_train.txt"
    kg_test = "../../../../Datasets/YAGO/YAGO43k/YAGO43k_name_test.txt"



(ent_id, rel_id) = extract_ent_rel_id([kg_train, kg_test])
(ent_rel_all, ent_rel_out, ent_rel_in) = extract_ent_rel_in_out(kg_train, ent_id, rel_id)
(rel_out_ent, rel_in_ent) = extract_rel_in_out_ent(kg_train, ent_id, rel_id)

N = len(ent_id)
M = len(rel_id)

data_SOP_test = extract_data_conv_id(kg_test, ent_id, rel_id)


total_test = len(data_SOP_test)

for s, o, p in data_SOP_test:
    union_rel = extract_union_rel(s, o, p, ent_rel_out, ent_rel_in)
    ent_s = extract_ent(p, rel_out_ent)
    ent_o = extract_ent(p, rel_in_ent)

    while len(union_rel) < hitsAtN:
        new_rel = randint(M)
        if new_rel not in union_rel:
            union_rel.append(new_rel)

    shuffle(union_rel)
    pred_rel = sample(union_rel, hitsAtN)

    while len(ent_s) < hitsAtN:
        new_ent = randint(N)
        if new_ent not in ent_s:
            ent_s.append(new_ent)

    shuffle(ent_s)
    pred_ent_s = sample(ent_s, hitsAtN)

    while len(ent_o) < hitsAtN:
        new_ent = randint(N)
        if new_ent not in ent_o:
            ent_o.append(new_ent)

    shuffle(ent_o)
    pred_ent_o = sample(ent_o, hitsAtN)



    if p in pred_rel:
        p_pos = pred_rel.index(p)

        for i in range(p_pos, len(pred_rel)):
            results_rel[i] += 1

    if s in pred_ent_s:
        s_pos = pred_ent_s.index(s)

        for i in range(s_pos, len(pred_ent_s)):
            results_ent[i] += 1

    if o in pred_ent_o:
        o_pos = pred_ent_o.index(o)

        for i in range(o_pos, len(pred_ent_o)):
            results_ent[i] += 1

print total_test
print results_rel
print results_ent

acc_rel = (results_rel / float(total_test)) * 100.
print acc_rel

acc_ent = (results_ent / float(total_test * 2)) * 100.
print acc_ent