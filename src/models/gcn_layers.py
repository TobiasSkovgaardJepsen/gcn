from mxnet.gluon import HybridBlock  # , Block
from mxnet.ndarray import array, eye
import numpy as np
from scipy.linalg import fractional_matrix_power


def power(matrix, fraction):
    return np.matrix(fractional_matrix_power(matrix, fraction))
# from mxnet.ndarray.linalg import gemm2 as matmul


class GraphConvolutionAggregatorBase(HybridBlock):
    def __init__(self, A, **kwargs):
        super().__init__(**kwargs)
        self.A = self.params.get_constant('A', A)


class SumAggregator(GraphConvolutionAggregatorBase):
    def hybrid_forward(self, F, X, A):
        return F.dot(A, X)  # Can be pre-computed for the first layer


class MeanAggregator(SumAggregator):
    def __init__(self, A, **kwargs):
        A = np.matrix(A.asnumpy())

        D = np.array(np.sum(A, axis=0))[0]
        D = np.matrix(np.diag(D))

        A = D**-1 * A
        A = array(A)
        super().__init__(A=A, **kwargs)


class GCNAggregator(SumAggregator):
    def __init__(self, A, **kwargs):
        # Missing self-loops
        A = np.matrix(A)

        D = np.array(np.sum(A, axis=0))[0]
        D = np.matrix(np.diag(D))

        A = power(D, -0.5) * A * power(D, -0.5)
        A = array(A)
        super().__init__(A=A, **kwargs)


class DenseAggregator(HybridBlock):
    def __init__(self, A, in_units, out_units, **kwargs):
        super().__init__(**kwargs)
        no_nodes = A.shape[0]
        with self.name_scope():
            self.W = self.params.get(
                'W', shape=(in_units, out_units))
            self.B = self.params.get(
                'B', shape=(no_nodes, out_units))

    def hybrid_forward(self, F, X, W, B):
        return F.dot(X, W)


class MaxPoolingAggregator(DenseAggregator):
    def hybrid_forward(self, F, X, W, B):
        dense = super().hybrid_forward(F, X, W, B)
        dense = dense.reshape((1, *dense.shape))
        print('DENSE', dense)
        pool = F.max(dense, axis=2)  # Node-wise max-pooling
        print('POOL', pool)
        reshape = pool.reshape((X.shape[0], 1))
        print('RESHAPE', reshape)
        return reshape


class MeanPoolingAggregator(DenseAggregator):
    def hybrid_forward(self, F, X, W, B):
        return F.mean(super().hybrid_forward(F, X, W, B))


class GraphConvBase(HybridBlock):
    def hybrid_forward(self, F, X):
        aggregation = self.aggregator(X)
        propagation = self.propegator(aggregation)
        return propagation


class Propagator(HybridBlock):
    def __init__(self, in_units, out_units,
                 activation=lambda x: x, **kwargs):
        super().__init__(**kwargs)

        self.activation = activation

        self.in_units = in_units
        self.out_units = out_units

        self.W = self.params.get(
            'W', shape=(self.in_units, self.out_units),
        )

    def hybrid_forward(self, F, X, W):
        return self.activation(
            F.dot(X, W))


class SimpleGraphConv(GraphConvBase):
    def __init__(self, A, in_units, out_units,
                 activation=lambda x: x, **kwargs):
        super().__init__(**kwargs)
        self.in_units = in_units
        self.out_units = out_units
        A = A.copy() + eye(*A.shape)

        with self.name_scope():
            self.aggregator = MeanAggregator(A)
            self.propegator = Propagator(
                self.in_units,
                self.out_units,
                activation)


class GraphConv(HybridBlock):
    def __init__(self, A, in_units, out_units,
                 activation=lambda x: x, **kwargs):
        super().__init__(**kwargs)

        self.activation = activation

        self.in_units = in_units
        self.out_units = out_units

        self.A = self.params.get_constant('A', A)
        self.W = self.params.get(
            'W', shape=(self.in_units, self.out_units),
        )
    """
    # Forward method for regular block
    def forward(self, X):
        aggregate = matmul(self.A.data(), X)
        propagate = self.activation(
            matmul(aggregate, self.W.data()))
        return propagate
    """

    def hybrid_forward(self, F, X, A, W):
        # Use symbolic API
        # F is symbol module
        # Remove references to self.<param> in body and take parameters
        # as input instead

        aggregate = F.dot(A, X)  # Can be pre-computed for the first layer
        propagate = self.activation(
            F.dot(aggregate, W))

        return propagate


class GraphConvSkip(HybridBlock):
    def __init__(self, A, in_units, out_units,
                 activation=lambda x: x, **kwargs):
        super().__init__(**kwargs)

        self.activation = activation

        self.in_units = in_units
        self.out_units = out_units

        self.A = self.params.get_constant('A', A)
        self.W = self.params.get(
            'W', shape=(
                2*self.in_units,
                self.out_units),
        )

    def hybrid_forward(self, F, X, A, W):
        # Use symbolic API
        # F is symbol module
        # Remove references to self.<param> in body and take parameters
        # as input instead
        aggregate = F.dot(A, X)
        concat = F.concat(X, aggregate)
        propagate = self.activation(
            F.dot(concat, W))

        return propagate


class GraphSageNormalization(GraphConvSkip):
    def hybrid_forward(self, F, X, **kwargs):
        unnormalized = super().hybrid_forward(F, X, **kwargs)
        return F.L2Normalization(unnormalized)
