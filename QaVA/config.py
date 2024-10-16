from QaVA.model import *
from QaVA.losses import union_weight_fn

class IndexConfig:
    def __init__(self):
        self.cache_root = 'cache'
        self.index_model = None
        self.query_objs = ['car']
        
        self.do_offline_indexing = True
        self.do_mining = True # Boolean that determines whether the mining step is skipped or not
        self.do_propagation = True
        self.do_training = True # Boolean that determines whether the training/fine-tuning step of the embedding dnn is skipped or not
        self.do_infer = True # Boolean that allows you to either compute embeddings or load them from cache
        
        self.batch_size = 16 # general batch size for both the target and embedding dnn
        self.nb_train = 7000 # controls how many datapoints are labeled to perform the triplet training
        self.train_lr = 1e-4
        self.max_training_epochs = int(1e6)
        self.early_stop = 5
        self.max_k = 5

        self.seed = 1
        self.val_rate = 0.1

        self.score_dnn = ScoreNet
        self.score_dnn_kwargs = {}
        self.encoder_config = {
            'otype': 'DenseGrid',
            'n_levels': 16,
            'n_features_per_level': 4,
            'log2_hashmap_size': 32,
            'base_resolution': 16,
            'per_level_scale': 1.35
        }
        self.mlp_config = {
            'otype': 'FullyFusedMLP',
            'activation': 'ReLU',
            'output_activation': 'None',
            'n_neurons': 64,
            'n_hidden_layers': 2
        }
        self.act_layer = 'None'

        self.weight_fn = union_weight_fn

    def eval(self, cache_root=None):
        if cache_root != None:
            self.cache_root = cache_root
        self.do_offline_indexing = False
        self.do_mining = False
        self.do_training = False
        self.do_infer = False
        return self