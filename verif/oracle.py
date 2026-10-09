from itertools import product

import numpy as np


def superset_sum(f, m):
    """Transformada de soma sobre superconjuntos, in-place: f[S] <- soma de f[T], T contém S."""
    idx = np.arange(f.size)
    for i in range(m):
        bit = 1 << i
        low = idx[(idx & bit) == 0]
        f[low] += f[low | bit]
    return f


class DomainOracle:
    
    def __init__(self, points, preds, m, name):
        self.points = np.asarray(points, dtype=int)
        self.preds = np.asarray(preds).astype(int)
        self.m = m
        self.name = name
        self._bits = 1 << np.arange(m)

    @property
    def n_points(self):
        return len(self.points)

    def waxp(self, v, c):
        v = np.asarray(v, dtype=int)
        agree = (self.points == v).astype(np.int64) @ self._bits
        size = 1 << self.m
        wrong = np.bincount(agree, weights=(self.preds != c).astype(float), minlength=size)
        superset_sum(wrong, self.m)
        return wrong == 0

    def counterexamples(self, v, c, mask):
        cols = [i for i in range(self.m) if (mask >> i) & 1]
        v = np.asarray(v, dtype=int)
        same = np.all(self.points[:, cols] == v[cols], axis=1) if cols else np.ones(len(self.points), bool)
        return self.points[same & (self.preds != c)]


def full_space(predict, domains, max_points=1 << 16):
    domains = [np.asarray(d) for d in domains]
    n_points = int(np.prod([len(d) for d in domains]))
    if n_points > max_points:
        raise ValueError(f"Espaço de entrada com {n_points} pontos excede o limite de {max_points}.")
    points = np.array(list(product(*domains)), dtype=int)
    return DomainOracle(points, predict(points), len(domains), "espaço completo")


def sample_space(predict, sample):
    points = np.unique(np.asarray(sample, dtype=int), axis=0)
    return DomainOracle(points, predict(points), points.shape[1], "dados de treino")
