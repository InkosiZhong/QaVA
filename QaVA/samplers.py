import math
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression

class Sampler(object):
    def __init__(self, err_tol, conf, Y_pred: np.ndarray, Y_true: np.ndarray, R, seed=0, masked_idx=None):
        self.err_tol = err_tol
        self.conf = conf
        self.R = R
        self.Y_pred, self.Y_true, self.idx, self.masked_val = self.permute(Y_pred.astype(np.float32).copy(), Y_true.copy(), seed, masked_idx)
        self.lb, self.ub = -10000000, 10000000
        self.beta = 1.5
        self.t, self.k = 1, 1
        self.p = 1.1
        self._c = self.conf * (self.p - 1) / self.p
        self.cov, self.c = None, None
        self.Xt_sum, self.Xt_sqsum = None, None
        self.Xt, self.x = None, None
        self.skip_sample = masked_idx is not None

    def get_sample(self, Y_pred, Y_true, nb_samples):
        raise NotImplementedError

    def reset(self, Y_pred, Y_true):
        pass
    
    def reestimate(self, Y_pred, Y_true, nb_samples):
        return None, None

    def permute(self, Y_pred, Y_true, seed, masked_idx):
        rand = np.random.RandomState(seed=seed)
        idx = rand.permutation(len(Y_pred))
        if masked_idx is not None:
            idx = np.setdiff1d(idx, masked_idx, assume_unique=True)
            masked_val = Y_true[masked_idx].sum()
        else:
            masked_val = 0
        Y_pred, Y_true = Y_pred[idx], Y_true[idx]
        return Y_pred, Y_true, idx, masked_val
    
    def update_param(self):
        alpha = np.floor(self.beta ** self.k) / np.floor(self.beta ** (self.k - 1))
        dk = self._c / (math.log(self.k, self.p) ** self.p)
        self.x = -alpha * np.log(dk) / 3
        t1, t2 = self.reestimate(self.Y_pred, self.Y_true, self.t)
        if t1 is not None and t2 is not None:
            self.Xt_sum = t1
            self.Xt_sqsum = t2

    def sample(self):
        self.t += 1
        if self.t > np.floor(self.beta ** self.k):
            self.k += 1
            self.update_param()

        sample = self.get_sample(self.Y_pred, self.Y_true, self.t)
        self.Xt_sum += sample
        self.Xt_sqsum += sample * sample
        self.Xt = self.Xt_sum / self.t
        sigmat = np.sqrt(1/self.t * (self.Xt_sqsum - self.Xt_sum ** 2 / self.t))
        # Finite sample correction
        sigmat *= np.sqrt((len(self.Y_true) - self.t) / (len(self.Y_true) - 1))

        ct = sigmat * np.sqrt(2 * self.x / self.t) + 3 * self.R * self.x / self.t
        self.lb = max(self.lb, np.abs(self.Xt) - ct)
        self.ub = min(self.ub, np.abs(self.Xt) + ct)
        return self.t
    
    def sync_sample(self, val):
        if self.skip_sample:
            self.skip_sample = False
            return
        self.Y_true[self.t + 1] = val
        _ = self.sample()
    
    def can_stop(self):
        return self.lb + self.err_tol >= self.ub - self.err_tol
    
    def get_result(self):
        estimate = np.sign(self.Xt) * 0.5 * \
            ((1 + self.err_tol) * self.lb + (1 - self.err_tol) * self.ub)
        return estimate * len(self.Y_true) + self.masked_val, self.t
    
    def get_sample_order(self):
        return self.idx


class ControlCovariateSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = np.mean(self.Y_pred)
        self.var_t = np.var(self.Y_pred)
        self.reset(self.Y_pred, self.Y_true)
        self.Xt_sum = self.get_sample(self.Y_pred, self.Y_true, self.t)
        self.Xt_sqsum = self.Xt_sum * self.Xt_sum

    def set_pred(self, Y_pred, R):
        self.R = R
        self.Y_pred[self.t + 1:] = Y_pred[self.idx][self.t + 1:]
        self.var_t = np.var(self.Y_pred)
        self.update_param()

    def reset(self, Y_pred, Y_true):
        self.cov = np.cov(Y_true[0:100].astype(np.float32), Y_pred[0:100].astype(np.float32))[0][1]
        self.c = -1 * self.cov / self.var_t

    def reestimate(self, Y_pred, Y_true, nb_samples):
        #yt_samp = Y_true[0:nb_samples]
        yt_samp = Y_true[0:nb_samples].astype(np.float32)
        yp_samp = Y_pred[0:nb_samples]
        self.cov = np.cov(yt_samp, yp_samp)[0][1]
        self.c = -1 * self.cov / self.var_t

        samples = yt_samp + self.c * (yp_samp - self.tau)
        Xt_sum = np.sum(samples)
        Xt_sqsum = sum([x * x for x in samples])
        return Xt_sum, Xt_sqsum

    def _get_yp(self, Y_true, Y_pred, nb_samples):
        return Y_pred[nb_samples]

    def get_sample(self, Y_pred, Y_true, nb_samples):
        Y_true[nb_samples] = float(Y_true[nb_samples])
        yt_samp = Y_true[nb_samples]
        yp_samp = self._get_yp(Y_true, Y_pred, nb_samples)

        sample = yt_samp + self.c * (yp_samp - self.tau)
        return sample