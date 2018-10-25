from mxnet.gluon import HybridBlock


class GraphConv(HybridBlock):
    def __init__(self, A, units,
                 in_units=0, activation=lambda x: x, **kwargs):
        super().__init__(**kwargs)

        self.A = self.params.get_constant('A', A)
        self.W = self.params('W', shape=(in_units, units))
        self.activation = activation

    def hybrid_forward(self, F, X):
        return self.activation(self.A * X * self.W)
