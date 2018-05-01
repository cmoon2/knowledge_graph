'''
Created on Jun 22, 2016

@author: Changsung Moon (cmoon2@ncsu.edu)
'''

import numpy as np


''' Unzip triples '''
def unzip_triples(xys, with_ys=False):
    xs, ys = list(zip(*xys))
    ss, os, ps = list(zip(*xs))
    if with_ys:
        return np.array(ss), np.array(ps), np.array(os), np.array(ys)
    else:
        return np.array(ss), np.array(ps), np.array(os)

''' Unzip triples of entities and entity types '''
def unzip_e_et(xys, with_ys=False):
    xs, ys = list(zip(*xys))
    es, ts = list(zip(*xs))
    if with_ys:
        return np.array(es), np.array(ts), np.array(ys)
    else:
        return np.array(es), np.array(ts)
    

''' Extract relation types for an entity '''
def extract_rel_from_ent(ent, ent_rel):
    list_rel = []

    if ent_rel.has_key(ent):
        list_rel = ent_rel[ent].keys()

    return list_rel


''' Extract union relation types between the subject and object '''
def extract_union_rel(s, o, p, ent_rel_out, ent_rel_in, except_p=False):
    union_rel = []
    count_p = 0

    if ent_rel_out.has_key(s) and ent_rel_in.has_key(o):
        union_rel = list(set(ent_rel_out[s].keys()) | set(ent_rel_in[o].keys()))
        if ent_rel_out[s].has_key(p):
            count_p += ent_rel_out[s][p]
        if ent_rel_in[o].has_key(p):
            count_p += ent_rel_in[o][p]
    elif ent_rel_out.has_key(s):
        union_rel = list(set(ent_rel_out[s].keys()))
        if ent_rel_out[s].has_key(p):
            count_p += ent_rel_out[s][p]
    elif ent_rel_in.has_key(o):
        union_rel = list(set(ent_rel_in[o].keys()))
        if ent_rel_in[o].has_key(p):
            count_p += ent_rel_in[o][p]

    if except_p == True and count_p == 1:
        union_rel.remove(p)

    return union_rel


''' Create pairs of positive triples and negative samples '''
def combine_pos_neg_union_pairs(pxs, nxs, ent_rel_out, ent_rel_in):
    output_sp = []
    output_pp = []
    output_op = []
    output_urp = []
    output_sn = []
    output_pn = []
    output_on = []
    output_urn = []

    sp, pp, op = unzip_triples(pxs)
    sn, pn, on = unzip_triples(nxs)

    for i in range(0, len(sp)):
        relations = []

        if pp[i] != pn[i]:
            relations = extract_union_rel(sp[i], op[i], pp[i], ent_rel_out, ent_rel_in, except_p=False)
        elif sp[i] != sn[i]:
            relations = extract_rel_from_ent(op[i], ent_rel_in)
        elif op[i] != on[i]:
            relations = extract_rel_from_ent(sp[i], ent_rel_out)

        for ur in relations:
            output_sp.append(sp[i])
            output_pp.append(pp[i])
            output_op.append(op[i])
            output_urp.append(ur)

            output_sn.append(sn[i])
            output_pn.append(pn[i])
            output_on.append(on[i])
            output_urn.append(ur)

    return np.array(output_sp), np.array(output_pp), np.array(output_op), np.array(output_urp), np.array(output_sn), np.array(output_pn), np.array(output_on), np.array(output_urn)






''' Convert triples into dictionary structure '''
def convert_triple_into_dict(data_SOP):
    triple_dict = {}

    for s, o, p in data_SOP:
        if triple_dict.has_key(s):
            if triple_dict[s].has_key(o):
                triple_dict[s][o].append(p)
            else:
                triple_dict[s][o] = [p]
        else:
            triple_dict[s] = {o: [p]}

    return triple_dict
