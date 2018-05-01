import numpy as np
#from skge_models.base_1 import Model
from skge_models.base_ete import Model
from skge_models.util import grad_sum_matrix

from func.util import unzip_e_et



class ETE(Model):
    """
    Translational Embeddings of Knowledge Graphs
    """

    def __init__(self, *args, **kwargs):
        super(ETE, self).__init__(*args, **kwargs)
        self.add_hyperparam('sz', args[0])
        self.add_hyperparam('ncomp', args[1])
        self.add_hyperparam('l1', kwargs.pop('l1', True))
        self.add_param('ET', (self.sz[1], self.ncomp))



    def _scores(self, es, ts, kg_model):

        score = kg_model.E[es] - self.ET[ts]


        if self.l1:
            score = np.abs(score)
        else:
            score = score ** 2


        return -np.sum(score, axis=1)




    def _pairwise_gradients(self, pxs, nxs, kg_model):
        ep, tp = unzip_e_et(pxs)
        # indices of negative tuples
        en, tn = unzip_e_et(nxs)

        pscores = self._scores(ep, tp, kg_model)
        nscores = self._scores(en, tn, kg_model)

        ind = np.where(nscores + self.margin > pscores)[0]

        # all examples in batch satify margin criterion
        self.nviolations = len(ind)

        if len(ind) == 0:
            return



        ep = list(ep[ind])
        en = list(en[ind])
        tp = list(tp[ind])
        tn = list(tn[ind])

        pg = self.ET[tp] - kg_model.E[ep]
        ng = self.ET[tn] - kg_model.E[en]


        if self.l1:
            pg = np.sign(-pg)
            ng = -np.sign(-ng)
        else:
            raise NotImplementedError()




        # entity type gradients
        ridx, Sm, n = grad_sum_matrix(tp + tn)
        gr = Sm.dot(np.vstack((-pg, -ng))) / n


        return {'ET': (gr, ridx)}
