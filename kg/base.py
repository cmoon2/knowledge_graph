'''
Original source code from https://github.com/mnick/holographic-embeddings

Modified by Changsung Moon (cmoon2@ncsu.edu) on Aug 16, 2016
'''


from __future__ import print_function
import argparse
import numpy as np
from numpy import argsort
from collections import defaultdict as ddict
import pickle
import timeit
import logging
import time


from func.io import extract_ent_rel_id
from func.io import extract_data_conv_id

from func.util import convert_triple_into_dict

from skge_models import sample
from skge_models.util import to_tensor

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('EX-KG')
np.random.seed(42)


class Experiment(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='Knowledge Graph experiment', conflict_handler='resolve')
        self.parser.add_argument('--margin', type=float, help='Margin for loss function')
        self.parser.add_argument('--init', type=str, default='nunif', help='Initialization method')
        self.parser.add_argument('--lr', type=float, help='Learning rate')
        self.parser.add_argument('--me', type=int, help='Maximum number of epochs')
        self.parser.add_argument('--ne', type=int, help='Numer of negative examples', default=1)
        self.parser.add_argument('--nb', type=int, help='Number of batches')
        self.parser.add_argument('--fout', type=str, help='Path to store model and results', default=None)
        self.parser.add_argument('--fin', type=str, help='Path to input data', default=None)
        self.parser.add_argument('--test-all', type=int, help='Evaluate Test set after x epochs', default=10)
        self.parser.add_argument('--no-pairwise', action='store_const', default=False, const=True)
        self.parser.add_argument('--mode', type=str, default='rank')
        self.parser.add_argument('--sampler', type=str, default='random-mode')
        self.neval = -1
        self.best_valid_score = -1.0
        self.exectimes = []
        self.ent_id = {}
        self.rel_id = {}
        self.id_ent = {}
        self.id_rel = {}

    def run(self):
        # parse comandline arguments
        self.args = self.parser.parse_args()

        if self.args.mode == 'rank':
            self.callback = self.ranking_callback
        elif self.args.mode == 'p_rank':
            self.callback = self.p_rank_callback
            self.evaluator = Rp_PredictionEval
        else:
            raise ValueError('Unknown experiment mode (%s)' % self.args.mode)
        self.train()

    def ranking_callback(self, trn, with_eval=False):
        # print basic info
        elapsed = timeit.default_timer() - trn.epoch_start
        self.exectimes.append(elapsed)
        if self.args.no_pairwise:
            log.info("[%3d] time = %ds, loss = %f" % (trn.epoch, elapsed, trn.loss))
        else:
            log.info("[%3d] time = %ds, violations = %d" % (trn.epoch, elapsed, trn.nviolations))

        # if we improved the validation error, store model and calc test error
        #if ((trn.epoch % self.args.test_all == 0) or with_eval) and (trn.epoch > 1800):
        if (trn.epoch % self.args.test_all == 0) or with_eval:
            start_time = time.time()  # Start time

            pos_v, fpos_v = self.ev_valid.positions(trn.model)
            fmrr_valid = ranking_scores(pos_v, fpos_v, trn.epoch, 'VALID')

            log.debug("FMRR valid = %f, best = %f" % (fmrr_valid, self.best_valid_score))

            #print("--- %s minutes ---" % ((time.time() - start_time) / 60))

            if fmrr_valid > self.best_valid_score:
                self.best_valid_score = fmrr_valid

                start_time = time.time()  # Start time

                pos_t, fpos_t = self.ev_test.positions(trn.model)
                ranking_scores(pos_t, fpos_t, trn.epoch, 'TEST')

                print("--- %s minutes ---" % ((time.time() - start_time) / 60))

                if self.args.fout is not None:
                    st = {
                        'model': trn.model,
                        'pos test': pos_t,
                        'fpos test': fpos_t,
                        'pos valid': pos_v,
                        'fpos valid': fpos_v,
                        'exectimes': self.exectimes,
                        'ent_id': self.ent_id,
                        'rel_id': self.rel_id,
                        'id_ent': self.id_ent,
                        'id_rel': self.id_rel
                    }
                    with open(self.args.fout, 'wb') as fout:
                        pickle.dump(st, fout, protocol=2)

        return True




    def p_rank_callback(self, trn, with_eval=False):
        # print basic info
        elapsed = timeit.default_timer() - trn.epoch_start
        self.exectimes.append(elapsed)
        if self.args.no_pairwise:
            log.info("[%3d] time = %ds, loss = %f" % (trn.epoch, elapsed, trn.loss))
        else:
            log.info("[%3d] time = %ds, violations = %d" % (trn.epoch, elapsed, trn.nviolations))

        # if we improved the validation error, store model and calc test error
        #if (trn.epoch % self.args.test_all == 0) or with_eval:
        if (trn.epoch == 1) or (trn.epoch % self.args.test_all == 1) or with_eval:
            start_time = time.time()  # Start time

            pos_v, fpos_v = self.ev_valid.positions(trn.model)
            mrr_valid = p_ranking_scores(pos_v, fpos_v, trn.epoch, 'VALID')

            log.debug("MRR valid = %f, best = %f" % (mrr_valid, self.best_valid_score))

            #print("--- %s minutes ---" % ((time.time() - start_time) / 60))

            if mrr_valid > self.best_valid_score:
                self.best_valid_score = mrr_valid

                start_time = time.time()  # Start time

                pos_t, fpos_t = self.ev_test.positions(trn.model)
                p_ranking_scores(pos_t, fpos_t, trn.epoch, 'TEST')

                print("--- %s minutes ---" % ((time.time() - start_time) / 60))

                if self.args.fout is not None:
                    st = {
                        'model': trn.model,
                        'pos test': pos_t,
                        'pos valid': pos_v,
                        'exectimes': self.exectimes,
                        'ent_id': self.ent_id,
                        'rel_id': self.rel_id,
                        'id_ent': self.id_ent,
                        'id_rel': self.id_rel
                    }
                    with open(self.args.fout, 'wb') as fout:
                        pickle.dump(st, fout, protocol=2)

        return True


    def train(self):
        (self.ent_id, self.rel_id, self.id_ent, self.id_rel) = extract_ent_rel_id([self.args.kg_train, self.args.kg_valid, self.args.kg_test])


        N = len(self.ent_id)
        M = len(self.rel_id)
        sz = (N, N, M)  # Sizes

        print("# of entities (N) = " + str(N))
        print("# of relations (M) = " + str(M))
        print("Size (sz) (N, N, M) = " + str(sz))

        data_SOP_train = extract_data_conv_id(self.args.kg_train, self.ent_id, self.rel_id)
        data_SOP_valid = extract_data_conv_id(self.args.kg_valid, self.ent_id, self.rel_id)
        data_SOP_test = extract_data_conv_id(self.args.kg_test, self.ent_id, self.rel_id)


        true_triples = data_SOP_train + data_SOP_test + data_SOP_valid
        if self.args.mode == 'rank':
            self.ev_test = self.evaluator(data_SOP_test, true_triples, self.neval)
            self.ev_valid = self.evaluator(data_SOP_valid, true_triples, self.neval)
        elif self.args.mode == 'p_rank':
            self.ev_test = self.evaluator(data_SOP_test, true_triples)
            self.ev_valid = self.evaluator(data_SOP_valid, true_triples)

        xs = data_SOP_train
        ys = np.ones(len(xs))

        # create sampling objects
        if self.args.sampler == 'corrupted':
            # create type index, here it is ok to use the whole data
            sampler = sample.CorruptedSampler(self.args.ne, xs, ti)
        elif self.args.sampler == 'random-mode':
            sampler = sample.RandomModeSampler(self.args.ne, [0, 1], xs, sz)
        elif self.args.sampler == 'lcwa':
            sampler = sample.LCWASampler(self.args.ne, [0, 1, 2], xs, sz)
        else:
            raise ValueError('Unknown sampler (%s)' % self.args.sampler)

        trn = self.setup_trainer(sz, sampler)
        log.info("Fitting model %s with trainer %s and parameters %s" % (
            trn.model.__class__.__name__,
            trn.__class__.__name__,
            self.args)
        )

        trn.fit(xs, ys)
        self.callback(trn, with_eval=True)


class FilteredRankingEval(object):

    def __init__(self, xs, true_triples, neval=-1):
        idx = ddict(list)
        tt = ddict(lambda: {'ss': ddict(list), 'os': ddict(list)})
        self.neval = neval
        self.sz = len(xs)
        for s, o, p in xs:
            idx[p].append((s, o))

        for s, o, p in true_triples:
            tt[p]['os'][s].append(o)
            tt[p]['ss'][o].append(s)

        self.idx = dict(idx)
        self.tt = dict(tt)

        self.neval = {}
        for p, sos in self.idx.items():
            if neval == -1:
                self.neval[p] = -1
            else:
                self.neval[p] = np.int(np.ceil(neval * len(sos) / len(xs)))

    def positions(self, mdl):
        pos = {}
        fpos = {}

        if hasattr(self, 'prepare_global'):
            self.prepare_global(mdl)

        for p, sos in self.idx.items():
            ppos = {'head': [], 'tail': []}
            pfpos = {'head': [], 'tail': []}

            if hasattr(self, 'prepare'):
                self.prepare(mdl, p)

            for s, o in sos[:self.neval[p]]:
                scores_o = self.scores_o(mdl, s, p).flatten()
                sortidx_o = argsort(scores_o)[::-1]
                ppos['tail'].append(np.where(sortidx_o == o)[0][0] + 1)

                rm_idx = self.tt[p]['os'][s]
                rm_idx = [i for i in rm_idx if i != o]
                scores_o[rm_idx] = -np.Inf
                sortidx_o = argsort(scores_o)[::-1]
                pfpos['tail'].append(np.where(sortidx_o == o)[0][0] + 1)

                scores_s = self.scores_s(mdl, o, p).flatten()
                sortidx_s = argsort(scores_s)[::-1]
                ppos['head'].append(np.where(sortidx_s == s)[0][0] + 1)

                rm_idx = self.tt[p]['ss'][o]
                rm_idx = [i for i in rm_idx if i != s]
                scores_s[rm_idx] = -np.Inf
                sortidx_s = argsort(scores_s)[::-1]
                pfpos['head'].append(np.where(sortidx_s == s)[0][0] + 1)
            pos[p] = ppos
            fpos[p] = pfpos

        return pos, fpos


class Rp_PredictionEval(object):

    def __init__(self, xs, true_triples):
        ss, os, ps = list(zip(*xs))
        self.ps = list(set(ps))
        self.xs = xs

        self.tt = convert_triple_into_dict(true_triples)


    def positions(self, mdl):
        pos = {}  # Raw Positions
        fpos = {}

        for s, o, p in self.xs:
            pos[p] = []
            fpos[p] = []

            scores_r = mdl._scores_r(s, o, p).flatten()
            sortidx_r = argsort(scores_r)[::-1]
            pos[p].append(np.where(sortidx_r == p)[0][0] + 1)

            rm_idx = self.tt[s][o]
            rm_idx = [i for i in rm_idx if i != p]
            scores_r[rm_idx] = -np.Inf
            sortidx_r = argsort(scores_r)[::-1]
            fpos[p].append(np.where(sortidx_r == p)[0][0] + 1)

        return pos, fpos


def ranking_scores(pos, fpos, epoch, txt):
    hpos = [p for k in pos.keys() for p in pos[k]['head']]
    tpos = [p for k in pos.keys() for p in pos[k]['tail']]
    fhpos = [p for k in fpos.keys() for p in fpos[k]['head']]
    ftpos = [p for k in fpos.keys() for p in fpos[k]['tail']]
    fmrr = _print_pos(
        np.array(hpos + tpos),
        np.array(fhpos + ftpos),
        epoch, txt)
    return fmrr


def p_ranking_scores(pos, fpos, epoch, txt):
    rpos = [p for k in pos.keys() for p in pos[k]]
    frpos = [p for k in fpos.keys() for p in fpos[k]]
    fmrr=_print_pos(
        np.array(rpos),
        np.array(frpos),
        epoch, txt)
    return fmrr


def _print_pos(pos, fpos, epoch, txt):
    mrr, mean_pos, hits = compute_scores(pos)
    fmrr, fmean_pos, fhits = compute_scores(fpos)
    log.info(
        "[%3d] %s: MRR = %.2f/%.2f, Mean Rank = %.2f/%.2f, Hits@1 = %.2f/%.2f, Hits@3 = %.2f/%.2f, Hits@10 = %.2f/%.2f" %
        (epoch, txt, mrr, fmrr, mean_pos, fmean_pos, hits[0], fhits[0], hits[1], fhits[1], hits[2], fhits[2])
    )
    return fmrr


def compute_scores(pos, hits=[1,3,10]):
    mrr = np.mean(1.0 / pos)
    mean_pos = np.mean(pos)
    hits_results = []
    for h in range(0, len(hits)):
        hits_results.append(np.mean(pos <= hits[h]).sum() * 100)
    return mrr, mean_pos, hits_results


def cardinalities(xs, ys, sz):
    T = to_tensor(xs, ys, sz)
    c_head = []
    c_tail = []
    for Ti in T:
        sh = Ti.tocsr().sum(axis=1)
        st = Ti.tocsc().sum(axis=0)
        c_head.append(sh[np.where(sh)].mean())
        c_tail.append(st[np.where(st)].mean())

    cards = {'1-1': [], '1-N': [], 'M-1': [], 'M-N': []}
    for k in range(sz[2]):
        if c_head[k] < 1.5 and c_tail[k] < 1.5:
            cards['1-1'].append(k)
        elif c_head[k] < 1.5:
            cards['1-N'].append(k)
        elif c_tail[k] < 1.5:
            cards['M-1'].append(k)
        else:
            cards['M-N'].append(k)
    return cards
