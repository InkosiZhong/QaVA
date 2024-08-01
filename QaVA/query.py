import QaVA
import sklearn
import numpy as np
import pandas as pd
import supg.datasource as datasource
from tqdm.autonotebook import tqdm
from QaVA.samplers import ControlCovariateSampler
from supg.sampler import ImportanceSampler
from supg.selector import ApproxQuery
from supg.selector import RecallSelector
#from supg.selector import ImportancePrecisionTwoStageSelector
from QaVA.selector import ImportancePrecisionTwoStageSelector
from tabulate import tabulate
import time

def print_dict(d, header='Key'):
    headers = [header, '']
    data = [(k,v) for k,v in d.items() if k not in ['y_preds', 'sample_idxs']]
    print(tabulate(data, headers=headers))

class BaseQuery:
    def __init__(self, index: QaVA.Index):
        self.index = index
        self.df = False
    
    def get_y_pred(self, scores):
        y_pred = scores.reshape(-1)
        r = max(1, np.amax(np.rint(y_pred)))
        return y_pred, r

    def score(self, target_dnn_output):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError

class AggregateQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError
    
    def _naive_execute(self, err_tol=0.01, confidence=0.05, seed=None):
        if seed == None:
            seed = np.random.randint(0, 6666)
        y_pred, r = self.get_y_pred(self.index.scores)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_pred))]
        )
        sampler = ControlCovariateSampler(err_tol, confidence, y_pred, y_true, r, seed=seed)
        sample_order = sampler.get_sample_order()
        while not sampler.can_stop():
            sampler.sample()
        estimate, nb_samples = sampler.get_result()
        res = {
            'initial_estimate': y_pred.sum(),
            'debiased_estimate': estimate,
            'nb_samples': nb_samples,
            'y_pred': y_pred,
            'y_true': y_true,
            'sample_idxs': sample_order[:nb_samples]
        }
        return res

    def _multi_execute(self, err_tol=0.01, confidence=0.05, seed=None, step=3000):
        if seed == None:
            seed = np.random.randint(0, 6666)
            
        # prop estimator
        y_prop, r = self.get_y_pred(self.index.prop_preds)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_prop))]
        )
        initial_estimates = [y_prop.sum()]
        y_preds = [y_prop]
        samplers = [ControlCovariateSampler(err_tol, confidence, y_prop, y_true, r, seed=seed)]
        sample_order = samplers[0].get_sample_order()
        print(f'[query] add a propagation-based sampler-0')

        # score net estimator
        y_pred, r = self.get_y_pred(self.index.scores)
        initial_estimates.append(y_pred.sum())
        y_preds.append(y_pred)
        samplers.append(ControlCovariateSampler(err_tol, confidence, y_pred, y_true, r, seed=seed))
        print(f'[query] add a scorenet-based sampler-1')
        estimate, nb_samples = None, None
        finished = [False, False]
        while estimate == None:
            for i, sampler in enumerate(samplers):
                if sampler.can_stop():
                    #estimate, _ = sampler.get_result()
                    initial_estimate = initial_estimates[i]
                    y_pred = y_preds[i]
                    if not finished[i]:
                        print(f'[query] sampler-{i} finish estimation at {sampler.t}')
                        finished[i] = True
                    #break
                if i == 0:
                    nb_samples = sampler.sample()
                    y_true[sample_order[nb_samples]] = sampler.Y_true[nb_samples]
                else: # Synchronization
                    sampler.sync_sample(y_true[sample_order[nb_samples]])
            if nb_samples % step == 0:
                sample_idxs = sample_order[:nb_samples]
                train_idxs = np.union1d(self.index.training_idxs, sample_idxs)
                train_labels = y_true[train_idxs]
                t = time.time()
                self.index.update(train_idxs, train_labels, finetune=False, prob_preds=y_prop)
                print(f'[query] add a new sampler-{len(samplers)} ({nb_samples}), cost {time.time() - t:.2f}s')
                y_pred, r = self.get_y_pred(self.index.scores)
                initial_estimates.append(y_pred.sum())
                y_preds.append(y_pred)
                samplers.append(ControlCovariateSampler(err_tol, confidence, y_pred, y_true, r, seed=seed, masked_idx=sample_idxs))
                finished.append(False)
        res = {
            'initial_estimate': initial_estimate,
            'debiased_estimate': estimate,
            'nb_samples': nb_samples,
            'y_pred': y_pred,
            'y_true': y_true,
            'y_preds': np.stack(y_preds, axis=0),
            'sample_idxs': sample_order[:nb_samples]
        }
        return res
    
    def _increase_execute(self, err_tol=0.01, confidence=0.05, seed=None, step=3000):
        if seed == None:
            seed = np.random.randint(0, 6666)
            
        # prop estimator
        y_pred, r = self.get_y_pred(self.index.prop_preds)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_pred))]
        )
        initial_estimates = [y_pred.sum()]
        y_preds = [y_pred]
        all_y_preds = [y_pred]
        samplers = [ControlCovariateSampler(err_tol, confidence, y_pred, y_true, r, seed=seed)]
        sample_order = samplers[0].get_sample_order()
        print(f'[query] add a propagation-based sampler-0')

        # score net estimator
        y_pred, r = self.get_y_pred(self.index.scores)
        initial_estimates.append(y_pred.sum())
        y_preds.append(y_pred)
        all_y_preds.append(y_pred)
        samplers.append(ControlCovariateSampler(err_tol, confidence, y_pred, y_true, r, seed=seed))
        print(f'[query] add a scorenet-based sampler-1')
        estimate, nb_samples = None, None
        lr = self.index.config.train_lr / 10
        while estimate == None:
            for i, sampler in enumerate(samplers):
                if sampler.can_stop():
                    estimate, nb_samples = sampler.get_result()
                    initial_estimate = initial_estimates[i]
                    y_pred = y_preds[i]
                    print(f'[query] sampler-{i} finish estimation')
                    break
                if i == 0:
                    nb_samples = sampler.sample()
                    y_true[sample_order[nb_samples]] = sampler.Y_true[nb_samples]
                else: # Synchronization
                    sampler.sync_sample(y_true[sample_order[nb_samples]])
            if nb_samples % step == 0:
                train_idxs = sample_order[nb_samples-step: nb_samples]
                train_labels = y_true[train_idxs]
                t = time.time()
                self.index.update(train_idxs, train_labels, finetune=True, prob_preds=y_pred, lr=lr)
                print(f'[query] finetune sampler-1 ({nb_samples}), cost {time.time() - t:.2f}s')
                y_pred, r = self.get_y_pred(self.index.scores)
                initial_estimates[-1] = y_pred.sum()
                y_preds[-1] = y_pred
                all_y_preds.append(y_pred)
                samplers[-1].set_pred(y_pred, r)

        res = {
            'initial_estimate': initial_estimate,
            'debiased_estimate': estimate,
            'nb_samples': nb_samples,
            'y_pred': y_pred,
            'y_true': y_true,
            'y_preds': np.stack(all_y_preds, axis=0),
            'sample_idxs': sample_order[:nb_samples]
        }
        return res

    def execute(self, err_tol=0.01, confidence=0.05, seed=None, step=3000, executer='increase'):
        if executer == 'naive':
            res = self._naive_execute(err_tol, confidence, seed)
        elif executer == 'multi':
            res = self._multi_execute(err_tol, confidence, seed, step)
        elif executer == 'increase':
            res = self._increase_execute(err_tol, confidence, seed, step)
        else:
            raise ValueError(f'executer={executer} is not supported')
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, err_tol=0.01, confidence=0.05, seed=None, step=3000, executer='increase'):
        if executer == 'naive':
            res = self._naive_execute(err_tol, confidence, seed)
        elif executer == 'multi':
            res = self._multi_execute(err_tol, confidence, seed, step)
        elif executer == 'increase':
            res = self._increase_execute(err_tol, confidence, seed, step)
        else:
            raise ValueError(f'executer={executer} is not supported')
        res['y_true'] = res['y_true'].astype(np.float32) # expensive
        res['actual_estimate'] = res['y_true'].sum()
        res['error'] = (res['debiased_estimate'] - res['actual_estimate']) / res['actual_estimate']
        print_dict(res, header=self.__class__.__name__)
        return res

class LimitQuery(BaseQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
    
    def _naive_execute(self, want_to_find=5, nb_to_find=10, GAP=300):
        y_pred, _ = self.get_y_pred(self.index.scores)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_pred))]
        )

        order = np.argsort(y_pred)[::-1]
        ret_inds = []
        visited = set()
        nb_calls = 0
        for ind in order:
            if ind in visited:
                continue
            nb_calls += 1
            if float(y_true[ind]) >= want_to_find:
                ret_inds.append(ind)
                for offset in range(-GAP, GAP+1):
                    visited.add(offset + ind)
            if len(ret_inds) >= nb_to_find:
                break
        res = {
            'nb_calls': nb_calls,
            'ret_inds': ret_inds,
            'y_pred': y_pred,
            'y_true': y_true
        }
        print_dict(res, header=self.__class__.__name__)
        return res
    
    def _multi_execute(self, want_to_find=5, nb_to_find=10, GAP=300, step=300, start_factor=10, min_sample=5, add_when_find=True):
        def weighted_pred(y_preds, weights):
            weights = np.array(weights, dtype=np.float32)
            weights /= weights.sum()
            return weights @ y_preds
        
        # prop estimator
        y_pred, _ = self.get_y_pred(self.index.prop_preds)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_pred))]
        )
        y_preds = y_pred
        weights = [self.index.config.nb_train]

        # score estimator
        y_pred, _ = self.get_y_pred(self.index.scores)
        y_preds = np.stack((y_preds, y_pred), axis=0)
        weights.append(self.index.config.nb_train)

        y_pred = weighted_pred(y_preds, weights)
        order = np.argsort(y_pred)[::-1]
        ret_inds = []
        visited = set()
        nb_calls = 0
        sample_idxs = []
        find = False
        while nb_calls < len(y_true):
            i = 0
            while nb_calls < len(y_true):
                if nb_calls < step: # random sampling
                    ind = order[i * start_factor]
                else:
                    ind = order[i]
                i += 1
                if ind in visited:
                    continue
                visited.add(ind)
                sample_idxs.append(ind)
                nb_calls += 1
                y_true[ind] = float(y_true[ind])
                if y_true[ind] >= want_to_find:
                    find = True
                    ret_inds.append(ind)
                    print('ret_inds:', len(ret_inds), 'nb_calls:', nb_calls, 'ind:', ind)
                    for offset in range(-GAP, GAP+1):
                        visited.add(offset + ind)
                if nb_calls % step == 0:
                    break
                if len(ret_inds) >= nb_to_find:
                    break
            if len(ret_inds) >= nb_to_find:
                break
            if len(ret_inds) < min_sample:
                continue
            if not add_when_find or find:
                find = False
                train_idxs = np.array(sample_idxs, dtype=int)
                train_idxs = np.union1d(self.index.training_idxs, train_idxs)
                train_labels = y_true[train_idxs].astype(float)
                t = time.time()
                self.index.update(train_idxs, train_labels, finetune=False, prob_preds=y_pred)
                print(f'[query] add new sampler ({nb_calls}), cost {time.time() - t:.2f}s')
                y_pred, _ = self.get_y_pred(self.index.scores)
                y_preds = np.concatenate((y_preds, y_pred.reshape(1, -1)), axis=0)
                weights.append(train_idxs.shape[0])
                y_pred = weighted_pred(y_preds, weights)
        res = {
            'nb_calls': nb_calls,
            'ret_inds': ret_inds,
            'y_pred': y_pred,
            'y_true': y_true,
            'sample_idxs': sample_idxs
        }
        print_dict(res, header=self.__class__.__name__)
        return res
    
    def _increase_execute(self, want_to_find=5, nb_to_find=10, GAP=300, step=300, start_factor=10, min_sample=5, add_when_find=True):
        y_pred, _ = self.get_y_pred(self.index.scores)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_pred))]
        )

        order = np.argsort(y_pred)[::-1]
        ret_inds = []
        visited = set()
        nb_calls = 0
        sample_idxs = []
        find = False
        while nb_calls < len(y_true):
            i = 0
            while nb_calls < len(y_true):
                if nb_calls < step: # random sampling
                    ind = order[i * start_factor]
                else:
                    ind = order[i]
                i += 1
                if ind in visited:
                    continue
                visited.add(ind)
                sample_idxs.append(ind)
                nb_calls += 1
                y_true[ind] = float(y_true[ind])
                if y_true[ind] >= want_to_find:
                    find = True
                    ret_inds.append(ind)
                    print('ret_inds:', len(ret_inds), 'nb_calls:', nb_calls, 'ind:', ind)
                    for offset in range(-GAP, GAP+1):
                        visited.add(offset + ind)
                if nb_calls % step == 0:
                    break
                if len(ret_inds) >= nb_to_find:
                    break
            if len(ret_inds) >= nb_to_find:
                break
            if len(ret_inds) < min_sample:
                continue
            if not add_when_find or find:
                find = False
                train_idxs = np.array(sample_idxs, dtype=int)
                train_labels = y_true[train_idxs].astype(float)
                t = time.time()
                self.index.update(train_idxs, train_labels, finetune=True, prob_preds=y_pred)
                print(f'[query] finetune sampler ({nb_calls}), cost {time.time() - t:.2f}s')
                y_pred, _ = self.get_y_pred(self.index.scores)
        res = {
            'nb_calls': nb_calls,
            'ret_inds': ret_inds,
            'y_pred': y_pred,
            'y_true': y_true,
            'sample_idxs': sample_idxs
        }
        print_dict(res, header=self.__class__.__name__)
        return res
    
    def execute(self, want_to_find=5, nb_to_find=10, GAP=300, step=300, start_factor=10, min_sample=10, add_when_find=True, executer='increase'):
        if executer == 'naive':
            res = self._naive_execute(want_to_find, nb_to_find, GAP)
        elif executer == 'multi':
            res = self._multi_execute(want_to_find, nb_to_find, GAP, step, start_factor, min_sample, add_when_find)
        elif executer == 'increase':
            res = self._increase_execute(want_to_find, nb_to_find, GAP, step, start_factor, min_sample, add_when_find)
        else:
            raise ValueError(f'executer={executer} is not supported')
        return res

    def execute_metrics(self, want_to_find=5, nb_to_find=10, GAP=300, step=300, start_factor=10, min_sample=10, add_when_find=True, executer='increase'):
        return self.execute(want_to_find, nb_to_find, GAP, step, start_factor, min_sample, add_when_find, executer)

class SUPGPrecisionQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError

    def _naive_execute(self, budget, min_precision, delta):
        y_pred, _ = self.get_y_pred(self.index.scores)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_pred))]
        )

        source = datasource.RealtimeDataSource(y_pred, y_true)
        sampler = ImportanceSampler()
        query = ApproxQuery(
            qtype='pt',
            min_recall=0.95, min_precision=min_precision, delta=delta,
            budget=budget
        )
        selector = ImportancePrecisionTwoStageSelector(query, source, sampler)
        inds = selector.select()

        res = {
            'inds': inds,
            'inds_length': inds.shape[0],
            'y_true': y_true,
            'y_pred': y_pred,
            'source': source
        }

        return res

    def _progressive_execute(self, budget, min_precision, delta):
        y_pred, _ = self.get_y_pred(self.index.scores)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_pred))]
        )

        query = ApproxQuery(
            qtype='pt',
            min_recall=0.95, min_precision=min_precision, delta=delta,
            budget=budget
        )
        # stage 1
        source1 = datasource.RealtimeDataSource(y_pred, y_true)
        sampler1 = ImportanceSampler()
        selector1 = ImportancePrecisionTwoStageSelector(query, source1, sampler1)
        cutoff_ub, samp1_ids = selector1.select_stage1()

        # stage 2
        t = time.time()
        samp_idxs = np.union1d(self.index.training_idxs, samp1_ids)
        self.index.update(samp_idxs, y_true[samp_idxs], finetune=False, prob_preds=None, logging=True)
        print(f'[query] finetune sampler ({len(samp_idxs)}), cost {time.time() - t:.2f}s')
        stage2_inds = source1.proxy_score_sort[:cutoff_ub]
        source2 = datasource.RealtimeDataSource(y_pred[stage2_inds], y_true[stage2_inds])
        sampler2 = ImportanceSampler()
        selector2 = ImportancePrecisionTwoStageSelector(query, source2, sampler2)
        set_inds, samp2_ids = selector2.select_stage2()
        set_inds, samp2_ids = stage2_inds[set_inds], stage2_inds[samp2_ids]
        samp_inds = source1.filter(np.concatenate([samp1_ids, samp2_ids]))
        inds = np.unique(np.concatenate([set_inds, samp_inds]))

        res = {
            'inds': inds,
            'inds_length': inds.shape[0],
            'y_true': y_true,
            'y_pred': y_pred,
            'source': source1
        }

        return res

    def execute(self, budget, min_precision, delta, executer='naive'):
        if executer == 'naive':
            res = self._naive_execute(budget, min_precision, delta)
        elif executer == 'progressive':
            res = self._progressive_execute(budget, min_precision, delta)
        else:
            raise ValueError(f'executer={executer} is not supported')
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, budget, min_precision, delta, executer='naive'):
        if executer == 'naive':
            res = self._naive_execute(budget, min_precision, delta)
        elif executer == 'progressive':
            res = self._progressive_execute(budget, min_precision, delta)
        else:
            raise ValueError(f'executer={executer} is not supported')
        source = res['source']
        inds = res['inds']
        nb_got = np.sum(source.lookup(inds))
        res['y_true'] = res['y_true'].astype(float)
        nb_true = res['y_true'].sum()
        precision = nb_got / len(inds)
        recall = nb_got / nb_true
        res['precision'] = precision
        res['recall'] = recall
        print_dict(res, header=self.__class__.__name__)
        return res

class SUPGRecallQuery(SUPGPrecisionQuery):
    def _execute(self, budget, min_recall, delta):
        #y_pred, _ = self.get_y_pred(self.index.scores)
        y_pred, _ = self.get_y_pred(self.index.prop_preds)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_pred))]
        )
        #y_pred = np.where(y_true.astype(float) > 0, 1, 0)

        source = datasource.RealtimeDataSource(y_pred, y_true)
        sampler = ImportanceSampler()
        query = ApproxQuery(
            qtype='rt',
            min_recall=min_recall, min_precision=0.90, delta=delta,
            budget=budget
        )
        selector = RecallSelector(query, source, sampler, sample_mode='sqrt')
        inds = selector.select()

        res = {
            'inds': inds,
            'inds_length': inds.shape[0],
            'y_true': y_true,
            'y_pred': y_pred,
            'source': source
        }
        return res

    def execute(self, budget, min_recall, delta):
        res = self._execute(budget, min_recall, delta)
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, budget, min_recall, delta):
        res = self._execute(budget, min_recall, delta)
        source = res['source']
        inds = res['inds']
        nb_got = np.sum(source.lookup(inds))
        res['y_true'] = res['y_true'].astype(float)
        nb_true = res['y_true'].sum()
        precision = nb_got / len(inds)
        recall = nb_got / nb_true
        res['precision'] = precision
        res['recall'] = recall
        print_dict(res, header=self.__class__.__name__)
        return res
    
from ExSample.chunk import Chunk
from ExSample.discriminator import Discriminator
from ExSample.sampler import Sampler
from ExSample.utils import Bbox, Object
import random
class WeightedChunk(Chunk):
    def __init__(self, start_id: int, length: int, weight: np.ndarray):
        super().__init__(start_id, length, False)
        self.weight = weight
        self.vis = []
        
    def sample(self) -> int:
        frame_id = -1
        data = [x for x in range(self.length) if x not in self.vis]
        weight = [x for i, x in enumerate(self.weight) if i not in self.vis]
        frame_id = random.choices(data, weights=weight, k=1)[0]
        self.vis.append(frame_id)
        self.nb_samples += 1
        return self.start_id + frame_id

    def update_weight(self, weight: np.ndarray):
        self.weight = weight
    
class ExSampleQuery(BaseQuery):
    def score(self, target_dnn_output):
        raise NotImplementedError

    def is_same_obj(self, frame_id1, det1, frame_id2, det2):
        raise NotImplementedError
    
    def _exsample_execute(self, nb_to_find=10000, nb_chunks=1024):
        discrim = Discriminator(self.is_same_obj, use_union_find=False)
        vid_len = len(self.index.scores)
        chunk_size = vid_len // nb_chunks
        chunks = []
        for i in range(nb_chunks):
            chunk_start = i * chunk_size
            if i == nb_chunks - 1:
                chunk_size = vid_len - chunk_start
            chunks.append(Chunk(chunk_start, chunk_size))
        sampler = Sampler(chunks=chunks, alpha0=0.1, beta0=1)
        ans = []
        ans_frame_id = []
        nb_samples = 0
        tbar = tqdm(total=nb_to_find, desc='Sampling')
        while len(ans) < nb_to_find and nb_samples < vid_len:
            # 1) choice of chunk and frame
            j, frame_id = sampler.sample()
            # 2) io,decode,detect,match
            frame = self.index.target_dnn_cache[frame_id]
            dets = [Object(
                    Bbox(d.xmin, d.ymin, d.xmax, d.ymax), d.object_name, d.confidence, d.ind, None) 
                for d in frame]
            # d0 are the unmatched dets
            # d1 are dets with only one match
            d0, d1, matches = discrim.get_matches(frame_id, dets)
            # 3) update state
            sampler.update_chunk_distribution(j, d0, d1)
            discrim.add(frame_id, dets, matches)
            ans += d0
            nb_samples += 1
            if len(d0) > 0:
                ans_frame_id.append(frame_id)
                tbar.update(len(d0))
            tbar.set_description(f'Sampling {nb_samples}')
        res = {
            'nb_calls': nb_samples,
            'nb_ret_objs': len(ans),
            'nb_ret_inds': len(ans_frame_id),
        }
        return res
    
    def _naive_execute(self, nb_to_find=10000, nb_chunks=1024):
        #y_pred, _ = self.get_y_pred(self.index.scores)
        y_pred, _ = self.get_y_pred(self.index.prop_preds)
        discrim = Discriminator(self.is_same_obj, use_union_find=False)
        vid_len = len(y_pred)
        chunk_size = vid_len // nb_chunks
        chunks = []
        for i in range(nb_chunks):
            chunk_start = i * chunk_size
            if i == nb_chunks - 1:
                chunk_size = vid_len - chunk_start
            weight = y_pred[chunk_start:chunk_start+chunk_size]
            weight_sum = weight.sum()
            if weight_sum == 0:
                weight = np.ones_like(weight) / chunk_size
            else:
                weight /= weight_sum
            chunks.append(WeightedChunk(chunk_start, chunk_size, weight))
            #chunks.append(Chunk(chunk_start, chunk_size))
        sampler = Sampler(chunks=chunks, alpha0=0.1, beta0=1)
        ans = []
        ans_frame_id = []
        nb_samples = 0
        tbar = tqdm(total=nb_to_find, desc='Sampling')
        while len(ans) < nb_to_find and nb_samples < vid_len:
            # 1) choice of chunk and frame
            j, frame_id = sampler.sample()
            # 2) io,decode,detect,match
            frame = self.index.target_dnn_cache[frame_id]
            dets = [Object(
                    Bbox(d.xmin, d.ymin, d.xmax, d.ymax), d.object_name, d.confidence, d.ind, None) 
                for d in frame]
            # d0 are the unmatched dets
            # d1 are dets with only one match
            d0, d1, matches = discrim.get_matches(frame_id, dets)
            # 3) update state
            sampler.update_chunk_distribution(j, d0, d1)
            discrim.add(frame_id, dets, matches)
            ans += d0
            nb_samples += 1
            if len(d0) > 0:
                ans_frame_id.append(frame_id)
                tbar.update(len(d0))
            tbar.set_description(f'Sampling {nb_samples}')
        res = {
            'nb_calls': nb_samples,
            'nb_ret_objs': len(ans),
            'nb_ret_inds': len(ans_frame_id),
        }
        return res
    
    def _increase_execute(self, nb_to_find=10000, nb_chunks=1024, step=1000):
        y_pred, _ = self.get_y_pred(self.index.scores)
        y_true = np.array(
            [QaVA.DNNOutputCacheFloat(self.index.target_dnn_cache, self.score, idx) for idx in range(len(y_pred))]
        )
        discrim = Discriminator(self.is_same_obj, use_union_find=False)
        vid_len = len(y_pred)
        chunk_size = vid_len // nb_chunks
        chunks = []
        for i in range(nb_chunks):
            chunk_start = i * chunk_size
            if i == nb_chunks - 1:
                chunk_size = vid_len - chunk_start
            weight = y_pred[chunk_start:chunk_start+chunk_size]
            weight_sum = weight.sum()
            if weight_sum == 0:
                weight = np.ones_like(weight) / chunk_size
            else:
                weight /= weight_sum
            chunks.append(WeightedChunk(chunk_start, chunk_size, weight))
        sampler = Sampler(chunks=chunks, alpha0=0.1, beta0=1)
        ans = []
        ans_frame_id = []
        train_idxs = []
        nb_samples = 0
        tbar = tqdm(total=nb_to_find, desc='Sampling')
        acc_step = step
        while len(ans) < nb_to_find and nb_samples < vid_len:
            # 1) choice of chunk and frame
            j, frame_id = sampler.sample()
            # 2) io,decode,detect,match
            frame = self.index.target_dnn_cache[frame_id]
            dets = [Object(
                    Bbox(d.xmin, d.ymin, d.xmax, d.ymax), d.object_name, d.confidence, d.ind, None) 
                for d in frame]
            # d0 are the unmatched dets
            # d1 are dets with only one match
            d0, d1, matches = discrim.get_matches(frame_id, dets)
            # 3) update state
            sampler.update_chunk_distribution(j, d0, d1)
            discrim.add(frame_id, dets, matches)
            ans += d0
            nb_samples += 1
            if len(d0) > 0:
                ans_frame_id.append(frame_id)
                train_idxs.append(frame_id)
                tbar.update(len(d0))
            tbar.set_description(f'Sampling {nb_samples}')
            # 4) finetune
            if len(ans) > acc_step:
                train_idxs = np.union1d(train_idxs, self.index.training_idxs)
                train_labels = y_true[train_idxs]
                self.index.update(train_idxs, train_labels, finetune=True, prob_preds=None)
                y_pred, _ = self.get_y_pred(self.index.scores)
                #y_pred += self.get_y_pred(self.index.scores)[0]
                chunk_size = vid_len // nb_chunks
                for i in range(nb_chunks):
                    chunk_start = i * chunk_size
                    if i == nb_chunks - 1:
                        chunk_size = vid_len - chunk_start
                    weight = y_pred[chunk_start:chunk_start+chunk_size]
                    weight_sum = weight.sum()
                    if weight_sum == 0:
                        weight = np.ones_like(weight) / chunk_size
                    else:
                        weight /= weight_sum
                    chunks[i].update_weight(weight)
                acc_step += step
                train_idxs = []
        res = {
            'nb_calls': nb_samples,
            'nb_ret_objs': len(ans),
            'nb_ret_inds': len(ans_frame_id),
        }
        return res
    
    def execute(self, nb_to_find, nb_chunks=1024, step=1000, executer='naive'):
        if executer == 'exsample':
            res = self._exsample_execute(nb_to_find, nb_chunks)
        elif executer == 'naive':
            res = self._naive_execute(nb_to_find, nb_chunks)
        elif executer == 'increase':
            res = self._increase_execute(nb_to_find, nb_chunks, step)
        print_dict(res, header=self.__class__.__name__)
        return res

    def execute_metrics(self, nb_to_find, nb_chunks=1024, step=1000, executer='naive'):
        return self.execute(nb_to_find, nb_chunks, step=step, executer=executer)