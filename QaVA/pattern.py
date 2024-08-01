import numpy as np
from typing import Dict, Callable
from QaVA.losses import union_weight_fn, head_prior_weight_fn, tail_prior_weight_fn, high_score_prior_weight_fn

class Footprint:
    def __init__(self, sample_idx: np.ndarray, all_label: np.ndarray, all_proxy_score: np.ndarray) -> None:
        self.sample_idx = sample_idx
        self.all_label = all_label
        self.all_proxy_score = all_proxy_score

    def get_distribution(self) -> Dict[int, float]:
        '''
        Construct the distribution from the footprint.
        '''
        sample_label = self.all_label[self.sample_idx]
        unqiue = np.unique(sample_label)
        dist = {x : sum(sample_label == x) / len(sample_label) for x in unqiue}
        return dist

class Pattern:
    def __init__(self, name: str, prior: int) -> None:
        self.name = name
        self.prior = prior
        
    def satisfy(self, footprint: Footprint) -> bool:
        raise NotImplementedError

    def get_loss_weight_fn(self) -> Callable:
        return union_weight_fn
    
class UnkownPattern(Pattern):
    def __init__(self) -> None:
        super().__init__('Unkown', 0)

    def satisfy(self, footprint: Footprint) -> bool:
        return True
    
class BinaryPattern(Pattern):
    def __init__(self) -> None:
        super().__init__('Binary', 4)
    
    def satisfy(self, footprint: Footprint) -> bool:
        sample_label = footprint.all_label[footprint.sample_idx]
        return (footprint.all_proxy_score >= 0).all() and \
                (footprint.all_proxy_score <= 1).all() and \
                (sample_label >= 0).all() and \
                (sample_label <= 1).all()
    
    def get_loss_weight_fn(self) -> Callable:
        return tail_prior_weight_fn

class TailPattern(Pattern):
    def __init__(self, tau: float=0.1, theta: float=1.25) -> None:
        super().__init__('Tail', 3)
        self.tau = tau
        self.theta = theta

    def satisfy(self, footprint: Footprint) -> bool:
        sample_label = footprint.all_label[footprint.sample_idx]
        dist = footprint.get_distribution()
        length = int(self.tau * len(sample_label))
        p1, p2 = 0, 0
        for x in sample_label[:length]:
            p1 += dist[x]
        for x in sample_label[length:]:
            p2 += dist[x]
        r = length / len(sample_label)
        return p1 / r * self.theta <= p2 / (1 - r)

    def get_loss_weight_fn(self) -> Callable:
        return tail_prior_weight_fn
    
class HighPattern(Pattern):
    def __init__(self, tau: int=0.1, theta: float=1.25) -> None:
        super().__init__('High', 2)
        self.tau = tau
        self.theta = theta

    def satisfy(self, footprint: Footprint) -> bool:
        sample_label = footprint.all_label[footprint.sample_idx]
        all_mean = np.mean(sample_label)
        length = int(self.tau * len(sample_label))
        first_mean = np.mean(sample_label[:length])
        return first_mean >= all_mean * self.theta

    def get_loss_weight_fn(self) -> Callable:
        return high_score_prior_weight_fn
    
class HeadPattern(Pattern):
    def __init__(self, n_seg: int=10, theta: float=1.25) -> None:
        super().__init__('Head', 1)
        self.n_seg = n_seg
        self.theta = theta

    def satisfy(self, footprint: Footprint) -> bool:
        sample_label = footprint.all_label[footprint.sample_idx]
        dist = footprint.get_distribution()
        length = int(len(sample_label) / self.n_seg)
        p = [sum([dist[x] for x in sample_label[i*length:(i+1)*length]]) 
                for i in range(self.n_seg)]
        return max(p) <= min(p) * self.theta

    def get_loss_weight_fn(self) -> Callable:
        return head_prior_weight_fn

class PatternManager:
    def __init__(self, k: int):
        self.patterns = [UnkownPattern()]
        self.patterns = sorted(self.patterns, key=lambda x: x.prior, reverse=True)
        self.query_history = {}
        self.k = k

    def add_pattern(self, pattern: Pattern):
        for p in self.patterns:
            if p.name == pattern.name:
                raise ValueError(f'Pattern {pattern.name} already exists.')
            if p.prior == pattern.prior:
                raise ValueError(f'Pattern {pattern.name} and {p.name} has the same priority {p.prior}.')
        self.patterns.append(pattern)
        self.patterns = sorted(self.patterns, key=lambda x: x.prior, reverse=True)
        # update
        for query_name, footprints in self.query_history.items():
            for i, (footprint, pattern_name) in enumerate(footprints):
                old_pattern = self.get_pattern(pattern_name)
                if pattern.prior > old_pattern.prior and pattern.satisfy(footprint):
                    self.query_history[query_name][i] = (footprint, pattern.name)

    def remove_pattern(self, name: str):
        assert name != 'Unkown', 'Cannot remove Unkown pattern.'
        for i, p in enumerate(self.patterns):
            if p.name == name:
                # update
                for query_name, footprints in self.query_history.items():
                    for i, (footprint, pattern_name) in enumerate(footprints):
                        if pattern_name == name:
                            pattern_name = self.select_pattern(footprint, i+1)
                            self.query_history[query_name][i] = (footprint, pattern_name)
                self.patterns.pop(i)
                return
        raise ValueError(f'Pattern {name} not found.')
    
    def get_pattern(self, name: str) -> Pattern:
        for p in self.patterns:
            if p.name == name:
                return p
        raise ValueError(f'Pattern {name} not found.')
    
    def select_pattern(self, footprint: Footprint, short_cut: int=0) -> str:
        for p in self.patterns[short_cut:]:
            if p.satisfy(footprint):
                print(f'[Pattern Manager] {p.name}({p.prior}) is selected.')
                return p.name
    
    def set_footprint(self, query_name: str, footprint: Footprint):
        pattern_name = self.select_pattern(footprint)
        if query_name in self.query_history:
            self.query_history[query_name].append((footprint, pattern_name))
        else:
            self.query_history[query_name] = [(footprint, pattern_name)]

    def vote_pattern(self, query_name: str) -> Pattern:
        vote = {'Unkown': 0}
        if query_name in self.query_history:
            for _, pattern_name in self.query_history[query_name][-self.k:]:
                if pattern_name in vote:
                    vote[pattern_name] += 1
                else:
                    vote[pattern_name] = 1
        pattern = None
        max_cnt = -1
        for p in self.patterns:
            if p.name in vote and vote[p.name] > max_cnt:
                max_cnt = vote[p.name]
                pattern = p
        print(f'[Pattern Manager] {pattern.name}({pattern.prior}) is voted')
        return pattern

if __name__ == '__main__':
    data = np.random.uniform(0, 1000, 500)
    hist, bin_edges = np.histogram(data, bins=10, range=(0, 1000))
    print(hist, bin_edges)