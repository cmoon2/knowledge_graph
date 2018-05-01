'''
Original source code from https://github.com/mnick/holographic-embeddings

Modified by Changsung Moon (cmoon2@ncsu.edu) on March 30, 2017
'''


#!/usr/bin/env python


#import sys
#sys.path.append("[PATH OF THE FOLDER OF SOURCE CODES]")
#print(sys.path)

import numpy as np
from base_conte_ete import Experiment, FilteredRankingEval_ETE
from skge_models import ETE, PairwiseStochasticTrainer_ETE



class ETE_Eval(FilteredRankingEval_ETE):

    def scores_t(self, mdl, kg_model, e, t):
        score = -np.sum(np.abs(mdl.ET - kg_model.E[e]), axis=1)
        
        return score





class ExpETE(Experiment):

    def __init__(self):
        super(ExpETE, self).__init__()
        self.parser.add_argument('--ncomp', type=int, help='Number of latent components', default=200)
        self.parser.add_argument('--margin', type=float, help='Margin for loss function', default=2.0)
        self.parser.add_argument('--init', type=str, default='nunif', help='Initialization method')
        self.parser.add_argument('--lr', type=float, help='Learning rate', default=0.1)
        self.parser.add_argument('--me', type=int, help='Maximum number of epochs', default=1000)
        self.parser.add_argument('--ne', type=int, help='Numer of negative examples', default=1)
        self.parser.add_argument('--nb', type=int, help='Number of batches', default=100)
        #self.parser.add_argument('--fout', type=str, help='Path to store model and results', default="../../models/ContE/ete_fb_3400epoch.model")
        #self.parser.add_argument('--fout', type=str, help='Path to store model and results', default=None)
        self.parser.add_argument('--fin', type=str, help='Path to input a model', default="../../../../../Documents/workspace/models/ContE/conte_fb_3400epoch.model")
        #self.parser.add_argument('--fin', type=str, help='Path to input a model', default="../../../../../Documents/workspace/models/ContE/rescal_et_yago_3500epoch.model")
        #self.parser.add_argument('--fin', type=str, help='Path to input a model', default="../../../../../Documents/workspace/models/ContE/conte_yago_name_2900epoch.model")

        self.parser.add_argument('--kg_train', type=str, help='Path to input train data', default="../../../../Datasets/Freebase/Missing_Entity_Types/FB15k_Missing_Entity_Type_train.txt")
        self.parser.add_argument('--kg_valid', type=str, help='Path to input valid data', default="../../../../Datasets/Freebase/Missing_Entity_Types/FB15k_Missing_Entity_Type_valid.txt")
        self.parser.add_argument('--kg_test', type=str, help='Path to input test data', default="../../../../Datasets/Freebase/Missing_Entity_Types/FB15k_Missing_Entity_Type_test.txt")

        #self.parser.add_argument('--kg_train', type=str, help='Path to input train data', default="../../../../Datasets/YAGO/Missing_Entity_Types/YAGO43k_Missing_Entity_Type_train.txt")
        #self.parser.add_argument('--kg_valid', type=str, help='Path to input valid data', default="../../../../Datasets/YAGO/Missing_Entity_Types/YAGO43k_Missing_Entity_Type_valid.txt")
        #self.parser.add_argument('--kg_test', type=str, help='Path to input test data', default="../../../../Datasets/YAGO/Missing_Entity_Types/YAGO43k_Missing_Entity_Type_test.txt")

        self.parser.add_argument('--test-all', type=int, help='Evaluate Test set after x epochs', default=50)
        self.parser.add_argument('--no-pairwise', action='store_const', default=False, const=True)
        
        ''' Entity Type Prediction'''
        self.parser.add_argument('--mode', type=str, default='et_rank')
        
        self.parser.add_argument('--sampler', type=str, default='random-mode')
        #self.parser.add_argument('--union_rel', type=str, default='out_in')
        self.evaluator = ETE_Eval

    def setup_trainer(self, sz, sampler, kg_model):
        model = ETE(sz, self.args.ncomp, init=self.args.init)
        trainer = PairwiseStochasticTrainer_ETE(
            model,
            kg_model,
            nbatches=self.args.nb,
            margin=self.args.margin,
            max_epochs=self.args.me,
            learning_rate=self.args.lr,
            samplef=sampler.sample,
            post_epoch=[self.callback]
        )
        return trainer

if __name__ == '__main__':
    ExpETE().run()
