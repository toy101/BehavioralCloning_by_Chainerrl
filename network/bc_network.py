import chainer
from chainer import functions as F
from chainer import links as L
from chainer import initializers
from chainerrl.distribution import SoftmaxDistribution
import numpy as np

class ContinuousBCNet(chainer.Chain):
    def __init__(self, action_space, hidden_layers=[100, 100, 100], seed=0):
        super().__init__()
        # assert hasattr(action_space, 'n')
        # assert hasattr(action_space, 'sample')
        self.action_size = action_space.shape[0]
        w_init = initializers.HeNormal(rng=np.random.RandomState(seed))

        with self.init_scope():
            self.hidden_layers = chainer.ChainList(*[L.Linear(None, h, initialW=w_init) for h in hidden_layers])
            # self.mu_layers = chainer.ChainList(*[L.Linear(None, self.action_size, initialW=w_init)])
            self.mu_layers = L.Linear(None, self.action_size, initialW=w_init)
            self.action_out = SoftmaxDistribution
            self.activation = F.relu

    def __call__(self, x):
        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h))

        mu_out = self.mu_layers(h)

        return self.action_out(mu_out)