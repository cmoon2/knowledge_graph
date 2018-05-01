'''
Original source code from https://github.com/mnick/scikit-kge

Modified by Changsung Moon (cmoon2@ncsu.edu) on Aug 16, 2016
'''


import numpy as np
from skge_models.base_conte import Model
from skge_models.util import grad_sum_matrix, unzip_triples
from skge_models.param import normalize

from func.util import combine_pos_neg_union_pairs



class ContE(Model):
    """
    Contextual KG Embeddings of Knowledge Graphs
    """

    def __init__(self, *args, **kwargs):
        super(ContE, self).__init__(*args, **kwargs)
        self.add_hyperparam('sz', args[0])
        self.add_hyperparam('ncomp', args[1])
        self.add_hyperparam('l1', kwargs.pop('l1', True))
        self.add_param('E', (self.sz[0], self.ncomp), post=normalize)
        self.add_param('R', (self.sz[2], self.ncomp))



    def _scores(self, ss, ps, os, urs):
        score = self.E[ss] + self.E[os] + self.R[urs] - self.R[ps]

        if self.l1:
            score = np.abs(score)
        else:
            score = score ** 2

        return -np.sum(score, axis=1)



    def _pairwise_gradients(self, pxs, nxs, mode):

        # indices of positive triples
        sp, pp, op = unzip_triples(pxs)
        # indices of negative triples
        sn, pn, on = unzip_triples(nxs)


        sp, pp, op, urp, sn, pn, on, urn = combine_pos_neg_union_pairs(pxs, nxs, self.ent_rel_out, self.ent_rel_in)

        pscores = self._scores(sp, pp, op, urp)
        nscores = self._scores(sn, pn, on, urn)

        '''
        if mode == "p_rank":

            sp, pp, op, urp = combine_SOP_union_RsRo(sp, pp, op, self.ent_rel_out, self.ent_rel_in)
            sn, pn, on, urn = combine_SOP_union_RsRo(sn, pn, on, self.ent_rel_out, self.ent_rel_in)

            pscores = self._scores(sp, pp, op, urp)
            nscores = self._scores(sn, pn, on, urn)

        elif mode == "rank":
            sp, pp, op, urp, sn, pn, on, urn = combine_pos_neg_union_pairs(pxs, nxs, self.ent_rel_out, self.ent_rel_in)

            pscores = self._scores(sp, pp, op, urp)
            nscores = self._scores(sn, pn, on, urn)
        '''

        ind = np.where(nscores + self.margin > pscores)[0]

        # all examples in batch satify margin criterion
        self.nviolations = len(ind)
        if len(ind) == 0:
            return


        sp = list(sp[ind])
        sn = list(sn[ind])
        pp = list(pp[ind])
        pn = list(pn[ind])
        op = list(op[ind])
        on = list(on[ind])

        urp = list(urp[ind])
        urn = list(urn[ind])


        pg = self.R[pp] - self.E[sp] - self.E[op] - self.R[urp]
        ng = self.R[pn] - self.E[sn] - self.E[on] - self.R[urn]



        if self.l1:
            pg = np.sign(-pg)
            ng = -np.sign(-ng)
        else:
            raise NotImplementedError()


        # entity gradients
        eidx, Sm, n = grad_sum_matrix(sp + op + sn + on)
        ge = Sm.dot(np.vstack((pg, pg, ng, ng))) / n

        # relation gradients
        ridx, Sm, n = grad_sum_matrix(pp + pn + urp + urn)
        gr = Sm.dot(np.vstack((-pg, -ng, pg, ng))) / n


        return {'E': (ge, eidx), 'R': (gr, ridx)}
