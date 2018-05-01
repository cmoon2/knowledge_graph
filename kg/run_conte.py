'''
Original source code from https://github.com/mnick/holographic-embeddings

Modified by Changsung Moon (cmoon2@ncsu.edu) on Aug 16, 2016
'''


#!/usr/bin/env python


#import sys
#sys.path.append("[PATH OF THE FOLDER OF SOURCE CODES]")
#print(sys.path)

import numpy as np
from base_conte_ete import Experiment, FilteredRankingEval
from skge_models import ContE, PairwiseStochasticTrainer_ContE

from func.util import extract_rel_from_ent



''' Evaluation '''
class ContEEval(FilteredRankingEval):

    def prepare_ER_so(self, mdl):
        self.ER_s = []
        self.ER_o = []

        for e in range(0, len(mdl.E)):
            Rs = extract_rel_from_ent(e, mdl.ent_rel_out)
            Ro = extract_rel_from_ent(e, mdl.ent_rel_in)

            if len(Rs) > 0:
                sum_Rs = np.sum(mdl.R[Rs], axis=0)
                self.ER_s.append(mdl.E[e] + (sum_Rs / float(len(Rs))))
            else:
                self.ER_s.append(mdl.E[e])

            if len(Ro) > 0:
                sum_Ro = np.sum(mdl.R[Ro], axis=0)
                self.ER_o.append(mdl.E[e] + (sum_Ro / float(len(Ro))))
            else:
                self.ER_o.append(mdl.E[e])

        self.ER_s = np.array(self.ER_s)
        self.ER_o = np.array(self.ER_o)


    def prepare(self, mdl, p):
        self.ER_s_p = self.ER_s - mdl.R[p]
        self.ER_o_p = self.ER_o - mdl.R[p]



    def scores_o(self, mdl, s, p):
        score = -np.sum(np.abs(mdl.E + self.ER_s_p[s]), axis=1)

        return score


    def scores_s(self, mdl, o, p):
        score = -np.sum(np.abs(mdl.E + self.ER_o_p[o]), axis=1)

        return score



''' Parameter Setup '''
class ExpContE(Experiment):

    def __init__(self):
        super(ExpContE, self).__init__()
        self.parser.add_argument('--ncomp', type=int, help='Number of latent components', default=200)
        self.parser.add_argument('--margin', type=float, help='Margin for loss function', default=2.0)
        self.parser.add_argument('--init', type=str, default='nunif', help='Initialization method')
        self.parser.add_argument('--lr', type=float, help='Learning rate', default=0.1)
        self.parser.add_argument('--me', type=int, help='Maximum number of epochs', default=3500)
        self.parser.add_argument('--ne', type=int, help='Numer of negative examples', default=1)
        self.parser.add_argument('--nb', type=int, help='Number of batches', default=100)
        #self.parser.add_argument('--fout', type=str, help='Path to store model and results', default="../../models/ContE/conte_fb_3400epoch.model")
        self.parser.add_argument('--fout', type=str, help='Path to store model and results', default=None)

        ''' Freebase Dataset '''
        #self.parser.add_argument('--kg_train', type=str, help='Path to input train data', default="../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-train.txt")
        #self.parser.add_argument('--kg_valid', type=str, help='Path to input valid data', default="../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-valid.txt")
        #self.parser.add_argument('--kg_test', type=str, help='Path to input test data', default="../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-test.txt")

        ''' YAGO Dataset '''
        self.parser.add_argument('--kg_train', type=str, help='Path to input train data', default="../../../../Datasets/YAGO/YAGO43k/YAGO43k_name_train.txt")
        self.parser.add_argument('--kg_valid', type=str, help='Path to input valid data', default="../../../../Datasets/YAGO/YAGO43k/YAGO43k_name_val.txt")
        self.parser.add_argument('--kg_test', type=str, help='Path to input test data', default="../../../../Datasets/YAGO/YAGO43k/YAGO43k_name_test.txt")

        self.parser.add_argument('--test-all', type=int, help='Evaluate Test set after x epochs', default=100)
        self.parser.add_argument('--no-pairwise', action='store_const', default=False, const=True)

        ''' Entity Prediction '''
        #self.parser.add_argument('--mode', type=str, default='rank')

        ''' Relation Type Prediction'''
        self.parser.add_argument('--mode', type=str, default='p_rank')

        self.parser.add_argument('--sampler', type=str, default='random-mode')
        self.evaluator = ContEEval

    def setup_trainer(self, sz, sampler):
        model = ContE(sz, self.args.ncomp, init=self.args.init)
        trainer = PairwiseStochasticTrainer_ContE(
            model,
            nbatches=self.args.nb,
            margin=self.args.margin,
            max_epochs=self.args.me,
            learning_rate=self.args.lr,
            samplef=sampler.sample,
            post_epoch=[self.callback]
        )
        return trainer

if __name__ == '__main__':
    ExpContE().run()
