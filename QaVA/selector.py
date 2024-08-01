from typing import Sequence

import numpy as np
import math

from supg.datasource import DataSource
from supg.sampler import Sampler, ImportanceSampler, SamplingBounds
from supg.selector.base_selector import BaseSelector, ApproxQuery


class ImportancePrecisionTwoStageSelector(BaseSelector):
    def __init__(
        self,
        query: ApproxQuery,
        data: DataSource,
        sampler: ImportanceSampler,
        start_samp=100,
        step_size=100
    ):
        self.query = query
        self.data = data
        self.sampler = sampler
        if not isinstance(sampler, ImportanceSampler):
            raise Exception("Invalid sampler for importance")
        self.start_samp = start_samp
        self.step_size = step_size
        self.n_sample_1 = self.query.budget // 2
        self.n_sample_2 = self.query.budget - self.n_sample_1
        self.data_idxs = self.data.get_ordered_idxs()
        self.n = len(self.data_idxs)
        self.x_basep = np.repeat((1./self.n),self.n)
        self.T = 1 + 2 * (self.query.budget - self.start_samp) // self.step_size

    def select_stage1(self):
        # TODO: weights
        x_probs = self.data.get_y_prob()
        # self.sampler.set_weights(np.repeat(1.,n)/n)
        self.sampler.set_weights(np.sqrt(x_probs))
        # self.sampler.set_weights(x_probs ** 2)
        # self.sampler.set_weights(x_probs)

        x_ranks = np.arange(self.n)
        x_weights = self.sampler.weights

        samp_ranks = np.sort(self.sampler.sample(max_idx=self.n, s=self.n_sample_1))
        samp_basep = self.x_basep[samp_ranks]
        samp_weights = x_weights[samp_ranks]
        samp1_ids = self.data_idxs[samp_ranks]
        samp_labels = self.data.lookup(samp1_ids)
        samp_masses = samp_basep / samp_weights

        delta = self.query.delta
        bounder = SamplingBounds(delta=delta / self.T)
        tpr_lb, tpr_ub = bounder.calc_bounds(
            fx = samp_labels*samp_masses,
        )
        cutoff_ub = int(math.ceil(tpr_ub * self.n / self.query.min_precision))
        print('cutoff ub: {}'.format(cutoff_ub))
        return cutoff_ub, samp1_ids

    def select_stage2(self, cutoff_ub=None):
        if cutoff_ub is None:
            cutoff_ub = self.n
        x_probs = self.data.get_y_prob()
        self.sampler.set_weights(np.sqrt(x_probs))
        samp2_ranks = np.sort(self.sampler.sample(max_idx=cutoff_ub, s=self.n_sample_2))
        x_weights = self.sampler.weights
        samp2_basep = self.x_basep[samp2_ranks]
        samp2_weights = x_weights[samp2_ranks]
        samp2_ids = self.data_idxs[samp2_ranks]
        samp2_labels = self.data.lookup(samp2_ids)
        samp2_masses = samp2_basep / samp2_weights
        # print("ns2: {}, len(samp2): {}".format(n_sample_2, len(samp2_ids)))

        delta = self.query.delta
        bounder = SamplingBounds(delta=delta / self.T)
        allowed = [0]
        for s_idx in range(self.start_samp, self.n_sample_2, self.step_size):
            if s_idx + 1 >= len(samp2_ranks):
                continue
            cur_u_idx = samp2_ranks[s_idx]
            # print("curidx: {}, s_idx: {}".format(cur_u_idx, s_idx))
            cur_x_basep = self.x_basep[:cur_u_idx+1] / np.sum(self.x_basep[:cur_u_idx+1])
            cur_x_weights = x_weights[:cur_u_idx+1] / np.sum(x_weights[:cur_u_idx+1])
            cur_x_masses = cur_x_basep / cur_x_weights

            cur_subsample_x_idxs = samp2_ranks[:s_idx+1]

            pos_rank_lb, pos_rank_ub = bounder.calc_bounds(
                fx=samp2_labels[:s_idx+1]*cur_x_masses[cur_subsample_x_idxs],
            )
            prec_lb = pos_rank_lb
            if prec_lb > self.query.min_precision:
                allowed.append(cur_u_idx)

        set_inds = self.data_idxs[:allowed[-1]]
        return set_inds, samp2_ids
    
    def reset_data(self, data: DataSource):
        self.data = data

    def select(self) -> Sequence:
        cutoff_ub, samp1_ids = self.select_stage1()
        set_inds, samp2_ids = self.select_stage2(cutoff_ub)
        samp_inds = self.data.filter(np.concatenate([samp1_ids, samp2_ids]))
        return np.unique(np.concatenate([set_inds, samp_inds]))