'''
Created on Jun 19, 2016

@author: Changsung Moon (cmoon2@ncsu.edu)
'''


import csv


''' Create IDs for entities and relation types '''
def extract_ent_rel_id(KG_files):
    ent_id = {}
    rel_id = {}
    id_ent = {}
    id_rel = {}

    ent_i = -1
    rel_i = -1

    for KG_file in KG_files:
        with open(KG_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter="\t")

            for r in reader:
                head = r[0]
                rel = r[1]
                tail = r[2]

                if not ent_id.has_key(head):
                    ent_i = ent_i + 1
                    ent_id[head] = ent_i
                    id_ent[ent_i] = head

                if not ent_id.has_key(tail):
                    ent_i = ent_i + 1
                    ent_id[tail] = ent_i
                    id_ent[ent_i] = tail

                if not rel_id.has_key(rel):
                    rel_i = rel_i + 1
                    rel_id[rel] = rel_i
                    id_rel[rel_i] = rel

    return (ent_id, rel_id, id_ent, id_rel)



''' Create IDs for entity types '''
def extract_et_id(KG_files):
    et_id = {}
    id_et = {}

    et_i = -1

    for KG_file in KG_files:
        with open(KG_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter="\t")

            for r in reader:
                type = r[1]

                if not et_id.has_key(type):
                    et_i = et_i + 1
                    et_id[type] = et_i
                    id_et[et_i] = type


    return et_id, id_et


''' Create datasets with IDs '''
def read_data_id(KG_file, ent_id, et_id):
    data_ET = []

    with open(KG_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")

        for r in reader:
            ent = r[0]
            type = r[1]

            data_ET.append((ent_id[ent], et_id[type]))

    return data_ET



''' Convert data into IDs '''
def extract_data_conv_id(KG_file, ent_id, rel_id):
    data_SOP = []

    with open(KG_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")

        for r in reader:
            head = r[0]
            rel = r[1]
            tail = r[2]

            data_SOP.append((ent_id[head], ent_id[tail], rel_id[rel]))

    return data_SOP


''' Extract relation types for entities '''
def extract_ent_rel_in_out(KG_file, ent_dict, rel_dict):
    ent_rel_all = {}
    ent_rel_out = {}
    ent_rel_in = {}

    with open(KG_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for r in reader:
            head = ent_dict[r[0]]
            rel = rel_dict[r[1]]
            tail = ent_dict[r[2]]

            if ent_rel_all.has_key(head):
                if ent_rel_all[head].has_key(rel):
                    ent_rel_all[head][rel] = ent_rel_all[head][rel] + 1
                else:
                    ent_rel_all[head][rel] = 1
            else:
                ent_rel_all[head] = {rel: 1}

            if ent_rel_all.has_key(tail):
                if ent_rel_all[tail].has_key(rel):
                    ent_rel_all[tail][rel] = ent_rel_all[tail][rel] + 1
                else:
                    ent_rel_all[tail][rel] = 1
            else:
                ent_rel_all[tail] = {rel: 1}

            if ent_rel_out.has_key(head):
                if ent_rel_out[head].has_key(rel):
                    ent_rel_out[head][rel] = ent_rel_out[head][rel] + 1
                else:
                    ent_rel_out[head][rel] = 1
            else:
                ent_rel_out[head] = {rel: 1}

            if ent_rel_in.has_key(tail):
                if ent_rel_in[tail].has_key(rel):
                    ent_rel_in[tail][rel] = ent_rel_in[tail][rel] + 1
                else:
                    ent_rel_in[tail][rel] = 1
            else:
                ent_rel_in[tail] = {rel: 1}



    return (ent_rel_all, ent_rel_out, ent_rel_in)



''' Extract entities for relation types '''
def extract_rel_in_out_ent(KG_file, ent_dict, rel_dict):
    rel_out_ent = {}
    rel_in_ent = {}

    with open(KG_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for r in reader:
            head = ent_dict[r[0]]
            rel = rel_dict[r[1]]
            tail = ent_dict[r[2]]

            if rel_out_ent.has_key(rel):
                if rel_out_ent[rel].has_key(head):
                    rel_out_ent[rel][head] += 1
                else:
                    rel_out_ent[rel][head] = 1
            else:
                rel_out_ent[rel] = {head: 1}

            if rel_in_ent.has_key(rel):
                if rel_in_ent[rel].has_key(tail):
                    rel_in_ent[rel][tail] += 1
                else:
                    rel_in_ent[rel][tail] = 1
            else:
                rel_in_ent[rel] = {tail: 1}

    return rel_out_ent, rel_in_ent