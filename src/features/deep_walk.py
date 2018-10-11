from gensim.models import Word2Vec
from random import shuffle, choice
from sklearn.base import BaseEstimator, TransformerMixin

def sample_random_walk(network, node, max_length):
    walk = [node]

    while len(walk) < max_length:
        neighbors = list(network.neighbors(
                walk[-1]))
        if not neighbors:
            break
        else:
            walk.append(
                choice(neighbors))

    return walk


def sample_random_walks(network, nodes, max_length):
    walks = []
    
    for node in nodes:
        walk = sample_random_walk(network, node, max_length)
        walks.append(walk)

    return walks


class DeepWalk(Word2Vec, BaseEstimator, TransformerMixin):
    def __init__(self, network, 
                 no_walks_per_node, max_walk_length,
                 dimensionality, context_size):
        self.network = network
        self.no_walks_per_node = no_walks_per_node
        self.max_walk_length = max_walk_length
        self.dimensionality = dimensionality
        self.context_size = context_size

        self.walks = self._sample_walks()

        super().__init__(
            sentences=[
                list(map(str, walk))
                for walk in self.walks],
            size=self.dimensionality,
            window=self.context_size,
            sg=1,  # Use SkipGram architecture
            hs=1,  # Use hierarchical softmax
            min_count=0,  # Keep all nodes
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [
            self.wv[str(node)]
            for node in X
        ]

    def _sample_walks(self):
        nodes = list(self.network.nodes())
        walks = []

        for _ in range(self.no_walks_per_node):
            shuffle(nodes)
            walks += sample_random_walks(
                self.network, nodes, self.max_walk_length
            )

        return walks
