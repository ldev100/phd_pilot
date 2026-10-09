from dataclasses import dataclass, field
from itertools import product

import numpy as np
from sklearn import datasets as skdatasets
from sklearn.feature_selection import mutual_info_classif


@dataclass
class Dataset:
    name: str
    X: np.ndarray             
    y: np.ndarray          
    domains: list  
    feature_names: list
    description: str
    origin: str = "sintética"
    class_names: dict = field(default_factory=lambda: {0: "classe 0", 1: "classe 1"})
    value_labels: list = None
    notes: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return self.X.shape[1]

    @property
    def space_size(self) -> int:
        return int(np.prod([len(d) for d in self.domains]))

    def describe_value(self, i, value):
        if self.value_labels is not None:
            return f"{self.feature_names[i]} na faixa {self.value_labels[i][int(value)]}"
        return f"{self.feature_names[i]} = {int(value)}"

    def value_label(self, i, value):
        if self.value_labels is not None:
            return self.value_labels[i][int(value)]
        return str(int(value))


def _all_points(domains):
    return np.array(list(product(*domains)), dtype=int)

def mux6(rng):
    """Multiplexador de 6 bits: 2 bits de endereço escolhem 1 de 4 bits de dados."""
    domains = [[0, 1]] * 6
    X = _all_points(domains)
    address = 2 * X[:, 0] + X[:, 1]
    y = X[np.arange(len(X)), 2 + address]
    X, y = np.vstack([X, X]), np.concatenate([y, y])
    names = ["a0", "a1", "d0", "d1", "d2", "d3"]
    return Dataset("mux6", X, y, domains, names, "Multiplexador de 6 bits")


def corral(rng, n=160):
    a0, a1, b0, b1, irr = (rng.integers(0, 2, n) for _ in range(5))
    y = ((a0 & a1) | (b0 & b1)).astype(int)
    corr = np.where(rng.random(n) < 0.75, y, 1 - y)
    X = np.column_stack([a0, a1, b0, b1, irr, corr])
    names = ["A0", "A1", "B0", "B1", "Irrelevante", "Correlacionada"]
    return Dataset("corral", X, y, [[0, 1]] * 6, names,
                   "(A0 e A1) ou (B0 e B1), com variável irrelevante e correlacionada")


def parity5_5(rng, n=1124):
   X = rng.integers(0, 2, size=(n, 10))
    y = X[:, :5].sum(axis=1) % 2
    names = [f"p{i}" for i in range(5)] + [f"r{i}" for i in range(5)]
    return Dataset("parity5+5", X, y, [[0, 1]] * 10, names,
                   "Paridade dos 5 primeiros bits; 5 bits irrelevantes")


MONK_DOMAINS = [[1, 2, 3], [1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2]]


def _monk(k, rng, noise=0.0):
    X = _all_points(MONK_DOMAINS)
    a1, a2, a3, a4, a5, a6 = X.T
    if k == 1:
        y = (a1 == a2) | (a5 == 1)
        desc = "(a1 = a2) ou (a5 = 1)"
    elif k == 2:
        y = (X == 1).sum(axis=1) == 2
        desc = "exatamente duas variáveis com o primeiro valor"
    else:
        y = ((a5 == 3) & (a4 == 1)) | ((a5 != 4) & (a2 != 3))
        desc = "(a5 = 3 e a4 = 1) ou (a5 != 4 e a2 != 3), com 5% de ruído"
    y = y.astype(int)
    if noise > 0:
        flip = rng.random(len(y)) < noise
        y = np.where(flip, 1 - y, y)
    names = [f"a{i}" for i in range(1, 7)]
    return Dataset(f"monk{k}", X, y, MONK_DOMAINS, names, desc)


def monk1(rng):
    return _monk(1, rng)


def monk2(rng):
    return _monk(2, rng)


def monk3(rng):
    return _monk(3, rng, noise=0.05)

PT_NAMES = {
    "mean radius": "raio (média)",
    "mean texture": "textura (média)",
    "mean perimeter": "perímetro (média)",
    "mean area": "área (média)",
    "mean compactness": "compacidade (média)",
    "mean concavity": "concavidade (média)",
    "mean concave points": "pontos côncavos (média)",
    "radius error": "raio (erro)",
    "perimeter error": "perímetro (erro)",
    "area error": "área (erro)",
    "worst radius": "raio (pior)",
    "worst texture": "textura (pior)",
    "worst perimeter": "perímetro (pior)",
    "worst area": "área (pior)",
    "worst compactness": "compacidade (pior)",
    "worst concavity": "concavidade (pior)",
    "worst concave points": "pontos côncavos (pior)",
    "alcohol": "teor alcoólico",
    "malic_acid": "ácido málico",
    "ash": "cinzas",
    "alcalinity_of_ash": "alcalinidade das cinzas",
    "magnesium": "magnésio",
    "total_phenols": "fenóis totais",
    "flavanoids": "flavonoides",
    "nonflavanoid_phenols": "fenóis não flavonoides",
    "proanthocyanins": "proantocianidinas",
    "color_intensity": "intensidade de cor",
    "hue": "matiz",
    "od280/od315_of_diluted_wines": "razão OD280/OD315",
    "proline": "prolina",
    "sepal length (cm)": "comprimento da sépala",
    "sepal width (cm)": "largura da sépala",
    "petal length (cm)": "comprimento da pétala",
    "petal width (cm)": "largura da pétala",
}

BIN_LABELS = { 
    2: ["baixa", "alta"],
    3: ["baixa", "intermediária", "alta"],
    4: ["muito baixa", "baixa", "alta", "muito alta"],
}


def _quantile_codes(col, n_bins):
    edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
    codes = np.searchsorted(edges, col, side="right")
    return codes.astype(int), list(range(len(edges) + 1))


def _real(name, data, y, k_features, n_bins, desc, class_names):
    X = np.asarray(data.data, dtype=float)
    mi = mutual_info_classif(X, y, random_state=0)
    top = np.sort(np.argsort(mi)[::-1][:k_features])
    cols, domains, labels = [], [], []
    for j in top:
        codes, dom = _quantile_codes(X[:, j], n_bins)
        cols.append(codes)
        domains.append(dom)
        labels.append(BIN_LABELS.get(len(dom), [str(v) for v in dom]))
    Xd = np.column_stack(cols)
    names = [PT_NAMES.get(str(data.feature_names[j]), str(data.feature_names[j])) for j in top]
    return Dataset(name, Xd, y.astype(int), domains, names, desc, origin="real (scikit-learn)",
                   class_names=class_names, value_labels=labels,
                   notes={"k_features": k_features, "n_bins": n_bins,
                          "original_names": [str(data.feature_names[j]) for j in top]})


def breast_cancer(rng):
    data = skdatasets.load_breast_cancer() 
    return _real("breast-cancer", data, data.target, 8, 3,
                 "Câncer de mama (Wisconsin): 8 variáveis, 3 faixas por quantis",
                 {0: "maligno", 1: "benigno"})


def wine(rng):
    data = skdatasets.load_wine()
    return _real("wine", data, (data.target == 0).astype(int), 8, 3,
                 "Vinhos: classe 0 contra as demais; 8 variáveis, 3 faixas",
                 {0: "outros cultivares", 1: "cultivar 0"})


def iris(rng):
    data = skdatasets.load_iris()
    return _real("iris", data, (data.target == 1).astype(int), 4, 4,
                 "Íris: versicolor contra as demais; 4 variáveis, 4 faixas",
                 {0: "outras espécies", 1: "versicolor"})


ALL = {
    "corral": corral,
    "mux6": mux6,
    "parity5+5": parity5_5,
    "monk1": monk1,
    "monk2": monk2,
    "monk3": monk3,
    "breast-cancer": breast_cancer,
    "wine": wine,
    "iris": iris,
}


def load_all(seed):
    rng = np.random.default_rng(seed)
    return [make(rng) for make in ALL.values()]
