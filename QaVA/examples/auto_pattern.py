from QaVA.examples.amsterdam import AmsterdamOfflineIndex, AmsterdamOfflineConfig
from QaVA.examples.video_db import AggregateQuery
from QaVA.pattern import *
from QaVA.utils import suppress_stdout_stderr

if __name__ == '__main__':
    pm = PatternManager(k=5)
    pm.add_pattern(BinaryPattern())
    pm.add_pattern(TailPattern(tau=0.1, theta=1.25))
    pm.add_pattern(HighPattern(tau=0.1, theta=1.25))
    pm.add_pattern(HeadPattern(n_seg=10, theta=1.05))

    config = AmsterdamOfflineConfig()
    config.act_layer = 'None'
    
    # 1st query
    pattern = pm.vote_pattern('aggregation')
    config.weight_fn = pattern.get_loss_weight_fn()
    config.query_objs = ['car']
    with suppress_stdout_stderr():
        index = AmsterdamOfflineIndex(config)
        index.init()
        query = AggregateQuery(index)
        res = query.execute_metrics(err_tol=0.01, confidence=0.05, seed=None, step=1000)
    footprint = Footprint(
        sample_idx=res['sample_idxs'],
        all_label=res['y_true'],
        all_proxy_score=res['y_pred'],
    )
    pm.set_footprint('aggregation', footprint)

    # 2nd query
    pattern = pm.vote_pattern('aggregation')