import QaVA
import numpy as np
import torch, torchvision
import os
from tqdm import tqdm
from numba import njit, jit, prange

"""
Both BlazeIt and SUPG assume for the sake of fast experiments that you have access to all of the Target DNN outputs.
These classes will allow you to still use the BlazeIt and SUPG algorithms by executing the Target DNN in realtime.
"""

class DNNOutputCache:
    def __init__(self, target_dnn, dataset, target_dnn_callback=lambda x: x):
        target_dnn.cuda()
        target_dnn.eval()
        self.target_dnn = target_dnn
        self.dataset = dataset
        self.target_dnn_callback = target_dnn_callback
        self.length = len(dataset)
        self.cache = [None]*self.length
        self.nb_of_invocations = 0
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.cache[idx] == None:
            with torch.no_grad():
                record = self.dataset[idx].unsqueeze(0).cuda()
                result = self.target_dnn(record)
            result = self.target_dnn_callback(result)
            self.cache[idx] = result
            self.nb_of_invocations += 1
        return self.cache[idx]
    
    def __setitem__(self, idx, value):
        self.cache[idx] = value
            
class DNNOutputCacheFloat:
    def __init__(self, target_dnn_cache, scoring_fn, idx):
        self.target_dnn_cache = target_dnn_cache
        self.scoring_fn = scoring_fn
        self.idx = idx
        
        def override_arithmetic_operator(name):
            def func(self, *args):
                value = self.target_dnn_cache[self.idx]
                value = self.scoring_fn(value)
                value = np.float32(value)
                args_f = []
                for arg in args:
                    if type(arg) is QaVA.utils.DNNOutputCacheFloat:
                        arg = np.float32(arg)
                    args_f.append(arg)
                value = getattr(value, name)(*args_f)
                return value 
            return func
        
        operator_names = [
            "__add__",
            "__sub__",
            "__mul__",
            "__truediv__", 
            "__neg__", 
            "__pos__", 
            "__radd__",
            "__rmul__",
        ]
            
        for name in operator_names:
            setattr(DNNOutputCacheFloat, name, override_arithmetic_operator(name))
        
    def __repr__(self):
        return f'DNNOutputCacheFloat(idx={self.idx})'
    
    def __float__(self):
        value = self.target_dnn_cache[self.idx]
        value = self.scoring_fn(value)
        return float(value)
    


@jit(parallel=True)
def _propagate(pred, topk_reps, topk_distances):
    '''
    numba acceleration
    topk_reps here is a sub-array of the true
    '''
    for i in prange(len(pred)):
        weights = topk_distances[i]
        weights = np.sum(weights) - weights
        weights = weights / weights.sum()
        counts = topk_reps[i]
        pred[i] =  np.sum(counts * weights)
    return pred

def propagate(target_dnn_cache, reps, topk_reps, topk_distances, score_fn):
    true = np.array(
        [DNNOutputCacheFloat(target_dnn_cache, score_fn, idx) for idx in range(len(topk_reps))]
    )
    pred = np.zeros(len(topk_reps))
    labels = np.zeros_like(reps)
    for i, rep in enumerate(reps):
        true[rep] = float(true[rep])
        labels[i] = true[rep]
    tmp = true[topk_reps]
    tmp = tmp.astype(np.float32)
    pred = _propagate(pred, tmp, topk_distances)
    return pred, labels


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

def eval_dnn_call(f, rounds, **param):
    cnt = 0
    t = tqdm(range(rounds), desc='querying')
    for i in t:
        with suppress_stdout_stderr():
            cnt += f(**param)['nb_samples']
        t.set_description(f'nb_samples={int(cnt/(i+1))}')
    return round(cnt / rounds)