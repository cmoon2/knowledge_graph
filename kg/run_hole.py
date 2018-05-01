'''
Original source code from https://github.com/mnick/holographic-embeddings

Modified by Changsung Moon (cmoon2@ncsu.edu) on May 30, 2016
'''


#!/usr/bin/env python

#import sys
#sys.path.append("[PATH OF THE FOLDER OF SOURCE CODES]")
#print(sys.path)


import numpy as np
from base import Experiment, FilteredRankingEval
from skge_models.util import ccorr
from skge_models import StochasticTrainer, PairwiseStochasticTrainer, HolE
from skge_models import activation_functions as afs


''' Evaluation '''
class HolEEval(FilteredRankingEval):

    def prepare(self, mdl, p):
        self.ER = ccorr(mdl.R[p], mdl.E)

    def scores_o(self, mdl, s, p):
        return np.dot(self.ER, mdl.E[s])

    def scores_s(self, mdl, o, p):
        return np.dot(mdl.E, self.ER[o])


''' Parameter Setup '''
class ExpHolE(Experiment):

    def __init__(self):
        super(ExpHolE, self).__init__()
        self.parser.add_argument('--rparam', type=float, help='Regularization for W', default=0)
        self.parser.add_argument('--afs', type=str, default='sigmoid', help='Activation function')
        self.parser.add_argument('--ncomp', type=int, help='Number of latent components', default=200)
        self.parser.add_argument('--margin', type=float, help='Margin for loss function', default=0.2)
        self.parser.add_argument('--init', type=str, default='nunif', help='Initialization method')
        self.parser.add_argument('--lr', type=float, help='Learning rate', default=0.1)
        self.parser.add_argument('--me', type=int, help='Maximum number of epochs', default=3500)
        self.parser.add_argument('--ne', type=int, help='Numer of negative examples', default=1)
        self.parser.add_argument('--nb', type=int, help='Number of batches', default=100)
        #self.parser.add_argument('--fout', type=str, help='Path to store model and results', default="../../models/ContE/hole_yago_name_2800epoch.model")
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
        self.evaluator = HolEEval

    def setup_trainer(self, sz, sampler):
        model = HolE(
            sz,
            self.args.ncomp,
            rparam=self.args.rparam,
            af=afs[self.args.afs],
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
    ExpHolE().run()
