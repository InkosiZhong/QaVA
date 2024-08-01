import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn
# float32 and 1D: cd thirdparty/tiny-cuda-nn-1d/bindings/torch && python setup.py install
# float16: pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

class TimeNet(nn.Module):
    def __init__(self, output_dim, encoder_config, mlp_config, embedding_size=512):
        nn.Module.__init__(self)
        self.encoder = tcnn.Encoding(n_input_dims=1, encoding_config=encoder_config)
        self.mlp = tcnn.Network(self.encoder.n_output_dims + embedding_size, output_dim, network_config=mlp_config)
        self.embedding_size = embedding_size

    def forward(self, t, embedding): # t in [0, 1]
        coord = self.encoder(t)
        return self.mlp(torch.cat((coord, embedding), dim=1))

class ConditionalNet(nn.Module):
    def __init__(self, output_dim, encoder_config, mlp_config, embedding_size=512, share_encoder=False):
        nn.Module.__init__(self)
        if share_encoder:
            self.encoder = tcnn.Encoding(n_input_dims=2, encoding_config=encoder_config)
            mlp_input_dims = self.encoder.n_output_dims + embedding_size
        else:
            self.t_encoder = tcnn.Encoding(n_input_dims=1, encoding_config=encoder_config)
            self.pred_encoder = tcnn.Encoding(n_input_dims=1, encoding_config=encoder_config)
            mlp_input_dims = self.t_encoder.n_output_dims + self.pred_encoder.n_output_dims + embedding_size
        self.mlp = tcnn.Network(mlp_input_dims, output_dim, network_config=mlp_config)
        self.share_encoder = share_encoder
    def forward(self, coord, embedding): # coord = (t, pred), t in [0, 1]
        if self.share_encoder:
            coord = self.encoder(coord)
            return self.mlp(torch.cat((coord, embedding), dim=1))
        else:
            t_coord, pred_coord = coord[:, :1], coord[:, 1:]
            t_coord = self.t_encoder(t_coord)
            pred_coord = self.pred_encoder(pred_coord)
            return self.mlp(torch.cat((t_coord, pred_coord, embedding), dim=1))

class RegressionNet:
    def __init__(self, unique_labels: np.ndarray) -> None:
        self.output_dim = 1
        self.min_val = unique_labels.min()
        self.scale = (unique_labels.max() - self.min_val)

    def scale_up(self, x):
        return x * self.scale + self.min_val

class ClassifyNet:
    def __init__(self, unique_labels: np.ndarray) -> None:
        self.unique_labels = torch.from_numpy(unique_labels)
        self.output_dim = unique_labels.shape[0]

    def get_label(self, x):
        return self.unique_labels[x]
    
def get_act_layer(name: str) -> nn.Module:
    layers = {
        'None': nn.Identity(),
        'Sigmoid': nn.Sigmoid(),
        'Softmax': nn.Softmax(dim=-1)
    }
    assert name in layers, f'act_layer={name} is invalid'
    return layers[name]

class ScoreNet(RegressionNet, TimeNet):
    def __init__(self, encoder_config, mlp_config, unique_labels, act_layer, embedding_size=512):
        RegressionNet.__init__(self, unique_labels)
        TimeNet.__init__(self, self.output_dim, encoder_config, mlp_config, embedding_size)
        self.act_layer = get_act_layer(act_layer)

    def forward(self, t, embedding): # t in [0, 1]
        out = self.scale_up(TimeNet.forward(self, t, embedding))
        return self.act_layer(out)
    
    def get_score(self, t, embedding):
        return self.forward(t, embedding)
        
class ConditionalScoreNet(RegressionNet, ConditionalNet):
    def __init__(self, encoder_config, mlp_config, unique_labels, act_layer, embedding_size=512, share_encoder=False):
        RegressionNet.__init__(self, unique_labels)
        ConditionalNet.__init__(self, self.output_dim, encoder_config, mlp_config, embedding_size, share_encoder)
        self.act_layer = get_act_layer(act_layer)

    def forward(self, coord, embedding, pred): # coord = (t, pred), t in [0, 1]
        out = ConditionalNet.forward(self, coord, embedding).view(-1) + pred
        return self.act_layer(out)
    
    def get_score(self, t, embedding, pred):
        return self.forward(t, embedding, pred)
    
class ScoreProbNet(ClassifyNet, TimeNet):
    def __init__(self, encoder_config, mlp_config, unique_labels, act_layer, embedding_size=512):
        ClassifyNet.__init__(self, unique_labels)
        TimeNet.__init__(self, self.output_dim, encoder_config, mlp_config, embedding_size)
        self.act_layer = get_act_layer(act_layer)
    
    def cuda(self):
        super().cuda()
        self.unique_labels = self.unique_labels.cuda()
        return self

    def forward(self, t, embedding): # t in [0, 1]
        out = TimeNet.forward(self, t, embedding)
        return self.act_layer(out)
    
    def get_score(self, t, embedding):
        prob = self.forward(t, embedding)
        return self.get_label(torch.argmax(prob, dim=-1))

class ConditionalScoreProbNet(ClassifyNet, ConditionalNet):
    def __init__(self, encoder_config, mlp_config, unique_labels, act_layer,
                 embedding_size=512, share_encoder=False):
        ClassifyNet.__init__(self, unique_labels)
        ConditionalNet.__init__(self, self.output_dim, encoder_config, mlp_config, embedding_size, share_encoder)
        self.act_layer = get_act_layer(act_layer)
            
    def cuda(self):
        super().cuda()
        self.unique_labels = self.unique_labels.cuda()
        return self

    def forward(self, coord, embedding, pred=None): # coord = (t, pred), t in [0, 1]
        out = ConditionalNet.forward(self, coord, embedding)
        return self.act_layer(out)
    
    def get_score(self, t, embedding, pred=None):
        prob = self.forward(t, embedding, pred)
        return prob