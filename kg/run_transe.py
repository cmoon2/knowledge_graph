'''
Original source code from https://github.com/mnick/holographic-embeddings

Modified by Changsung Moon (cmoon2@ncsu.edu) on Jul 9, 2016
'''



#!/usr/bin/env python

#import sys
#sys.path.append("[PATH OF THE FOLDER OF SOURCE CODES]")
#print(sys.path)


import numpy as np
from base import Experiment, FilteredRankingEval
from skge_models import TransE, PairwiseStochasticTrainer


''' Evaluation '''
class TransEEval(FilteredRankingEval):

    def prepare(self, mdl, p):
        self.ER = mdl.E + mdl.R[p]

    def scores_o(self, mdl, s, p):
        return -np.sum(np.abs(self.ER[s] - mdl.E), axis=1)

    def scores_s(self, mdl, o, p):
        return -np.sum(np.abs(self.ER - mdl.E[o]), axis=1)


''' Parameter Setup '''
class ExpTransE(Experiment):

    def __init__(self):
        super(ExpTransE, self).__init__()
        self.parser.add_argument('--ncomp', type=int, help='Number of latent components', default=100)
        self.parser.add_argument('--margin', type=float, help='Margin for loss function', default=2.0)
        self.parser.add_argument('--init', type=str, default='nunif', help='Initialization method')
        self.parser.add_argument('--lr', type=float, help='Learning rate', default=0.1)
        self.parser.add_argument('--me', type=int, help='Maximum number of epochs', default=200)
        self.parser.add_argument('--ne', type=int, help='Numer of negative examples', default=1)
        self.parser.add_argument('--nb', type=int, help='Number of batches', default=100)
        #self.parser.add_argument('--fout', type=str, help='Path to store model and results', default="../../models/ContE/transe_yago_name_d100_3500epoch.model")
        self.parser.add_argument('--fout', type=str, help='Path to store model and results', default=None)

        ''' Freebase Dataset '''
        self.parser.add_argument('--kg_train', type=str, help='Path to input train data', default="../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-train.txt")
        self.parser.add_argument('--kg_valid', type=str, help='Path to input valid data', default="../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-valid.txt")
        self.parser.add_argument('--kg_test', type=str, help='Path to input test data', default="../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-test.txt")

        ''' YAGO Dataset '''
        #self.parser.add_argument('--kg_train', type=str, help='Path to input train data', default="../../../../Datasets/YAGO/YAGO43k/YAGO43k_name_train.txt")
        #self.parser.add_argument('--kg_valid', type=str, help='Path to input valid data', default="../../../../Datasets/YAGO/YAGO43k/YAGO43k_name_val.txt")
        #self.parser.add_argument('--kg_test', type=str, help='Path to input test data', default="../../../../Datasets/YAGO/YAGO43k/YAGO43k_name_test.txt")

        self.parser.add_argument('--test-all', type=int, help='Evaluate Test set after x epochs', default=3)
        self.parser.add_argument('--no-pairwise', action='store_const', default=False, const=True)

        ''' Entity Prediction '''
        #self.parser.add_argument('--mode', type=str, default='rank')

        ''' Relation Type Prediction'''
        self.parser.add_argument('--mode', type=str, default='p_rank')

        self.parser.add_argument('--sampler', type=str, default='random-mode')
        self.evaluator = TransEEval

    def setup_trainer(self, sz, sampler):
        model = TransE(sz, self.args.ncomp, init=self.args.init)
        trainer = PairwiseStochasticTrainer(
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
    ExpTransE().run()
