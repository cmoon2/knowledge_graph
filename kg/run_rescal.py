'''
Original source code from https://github.com/mnick/holographic-embeddings

Modified by Changsung Moon (cmoon2@ncsu.edu) on Jul 9, 2016
'''


#!/usr/bin/env python

#import sys
#sys.path.append("[PATH OF THE FOLDER OF SOURCE CODES]")
#print(sys.path)


import numpy as np
from numpy import dot
from base import Experiment, FilteredRankingEval
from skge_models import StochasticTrainer, PairwiseStochasticTrainer, RESCAL


''' Evaluation '''
class RESCALEval(FilteredRankingEval):

    def prepare(self, mdl, p):
        return 0

    def scores_o(self, mdl, s, p):

        return np.array([
            dot(mdl.E[s], dot(mdl.W[p], mdl.E[i]))
            for i in range(len(mdl.E))
        ])

    def scores_s(self, mdl, o, p):

        return np.array([
            dot(mdl.E[i], dot(mdl.W[p], mdl.E[o]))
            for i in range(len(mdl.E))
        ])


''' Parameter Setup '''
class ExpRESCAL(Experiment):

    def __init__(self):
        super(ExpRESCAL, self).__init__()
        self.parser.add_argument('--rparam', type=float, help='Regularization for W', default=0)
        #self.parser.add_argument('--afs', type=str, default='sigmoid', help='Activation function')
        self.parser.add_argument('--ncomp', type=int, help='Number of latent components', default=150)
        self.parser.add_argument('--margin', type=float, help='Margin for loss function', default=0.2)
        self.parser.add_argument('--init', type=str, default='nunif', help='Initialization method')
        self.parser.add_argument('--lr', type=float, help='Learning rate', default=0.1)
        self.parser.add_argument('--me', type=int, help='Maximum number of epochs', default=3500)
        self.parser.add_argument('--ne', type=int, help='Numer of negative examples', default=1)
        self.parser.add_argument('--nb', type=int, help='Number of batches', default=100)
        #self.parser.add_argument('--fout', type=str, help='Path to store model and results', default="../../models/ContE/rescal_yago_3300epoch.model")
        self.parser.add_argument('--fout', type=str, help='Path to store model and results', default=None)

        ''' Freebase Dataset '''
        self.parser.add_argument('--kg_train', type=str, help='Path to input train data', default="../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-train.txt")
        self.parser.add_argument('--kg_valid', type=str, help='Path to input valid data', default="../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-valid.txt")
        self.parser.add_argument('--kg_test', type=str, help='Path to input test data', default="../../../../Datasets/Freebase/FB15k/freebase_mtr100_mte100-test.txt")

        ''' YAGO Dataset '''
        #self.parser.add_argument('--kg_train', type=str, help='Path to input train data', default="../../../../Datasets/YAGO/YAGO43k/YAGO42k_name_train.txt")
        #self.parser.add_argument('--kg_valid', type=str, help='Path to input valid data', default="../../../../Datasets/YAGO/YAGO43k/YAGO42k_name_val.txt")
        #self.parser.add_argument('--kg_test', type=str, help='Path to input test data', default="../../../../Datasets/YAGO/YAGO43k/YAGO43k_name_test.txt")

        self.parser.add_argument('--test-all', type=int, help='Evaluate Test set after x epochs', default=100)
        self.parser.add_argument('--no-pairwise', action='store_const', default=False, const=True)

        ''' Entity Prediction '''
        #self.parser.add_argument('--mode', type=str, default='rank')

        ''' Relation Type Prediction'''
        self.parser.add_argument('--mode', type=str, default='p_rank')

        self.parser.add_argument('--sampler', type=str, default='random-mode')
        self.evaluator = RESCALEval

    def setup_trainer(self, sz, sampler):
        model = RESCAL(
            sz,
            self.args.ncomp,
            rparam=self.args.rparam,
            #af=afs[self.args.afs],
            init=self.args.init
        )
        if self.args.no_pairwise:
            trainer = StochasticTrainer(
                model,
                nbatches=self.args.nb,
                max_epochs=self.args.me,
                post_epoch=[self.callback],
                learning_rate=self.args.lr,
                samplef=sampler.sample
            )
        else:
            trainer = PairwiseStochasticTrainer(
                model,
                nbatches=self.args.nb,
                max_epochs=self.args.me,
                post_epoch=[self.callback],
                learning_rate=self.args.lr,
                margin=self.args.margin,
                samplef=sampler.sample
            )
        return trainer

if __name__ == '__main__':
    ExpRESCAL().run()
