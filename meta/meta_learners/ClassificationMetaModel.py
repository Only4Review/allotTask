"""
As in https://github.com/tristandeleu/pytorch-meta/blob/master/examples/maml/model.py
"""

import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)


def conv3x3(in_channels, out_channels, kernel_size = 3, pooling = 2, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2), **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(pooling)
    )

class ConvMetaModel(MetaModule):
    def __init__(self, in_channels, out_features, hidden_sizes=[64] * 4, kernel_sizes=[3] * 4, pooling = [2]*4):
        super(ConvMetaModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_sizes[0], kernel_sizes[0], pooling[0]),
            conv3x3(hidden_sizes[0], hidden_sizes[1], kernel_sizes[1], pooling[1]),
            conv3x3(hidden_sizes[1], hidden_sizes[2], kernel_sizes[2], pooling[2]),
            conv3x3(hidden_sizes[2], hidden_sizes[3], kernel_sizes[3], pooling[3])
        )

        self.classifier = MetaLinear(256, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits
        
class ConvMetaModelImagenet(MetaModule):
    def __init__(self, in_channels, out_features, hidden_sizes=[32] * 4, kernel_sizes=[3] * 4, pooling = [2]*4): ## Imagenet
        super(ConvMetaModelImagenet, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_sizes[0], kernel_sizes[0], pooling[0]),
            conv3x3(hidden_sizes[0], hidden_sizes[1], kernel_sizes[1], pooling[1]),
            conv3x3(hidden_sizes[1], hidden_sizes[2], kernel_sizes[2], pooling[2]),
            conv3x3(hidden_sizes[2], hidden_sizes[3], kernel_sizes[3], pooling[3])
        )

        self.classifier = MetaLinear(800, out_features)
      
    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features')); #print(features.size())
        features = features.view((features.size(0), -1)); #print(features.size())
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits  
