from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from verif import datasets
from verif.claims import Facts
from verif.oracle import full_space, sample_space
from verif.shap_tool import shap_for_predicted

SEED = 2027
MODELS = {
    "AD": lambda: DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "RF": lambda: RandomForestClassifier(n_estimators=50, max_depth=5,
                                         random_state=SEED, n_jobs=1),
    "GB": lambda: GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                             random_state=SEED),
}
REAL = ("breast-cancer", "wine", "iris")


def split_and_select(ds, instances, seed=SEED):
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=0.3, random_state=seed, stratify=ds.y)
    rng = np.random.default_rng(seed)
    unique_test = np.unique(X_te, axis=0)
    explain = unique_test[rng.permutation(len(unique_test))[:instances]]
    background = X_tr[rng.permutation(len(X_tr))[:100]]
    return X_tr, X_te, y_tr, y_te, explain, background


@dataclass
class Case:
    ds: object               # datasets.Dataset
    modelo: str              # "AD", "RF" ou "GB"
    instancia: int           # posição na lista de instâncias explicadas da base
    v: np.ndarray
    c: int
    phi: np.ndarray          # contribuições SHAP orientadas à classe predita
    p_pred: float            # probabilidade estimada da classe predita
    f_full: Facts            # fatos no espaço de entrada completo
    f_data: Facts            # fatos nos dados de treino
    full: object             # oráculos (para contraexemplos)
    data: object
    acuracia: float

    @property
    def label(self):
        return self.ds.class_names[self.c]

    @property
    def other_label(self):
        return self.ds.class_names[1 - self.c]

    @property
    def key(self):
        return f"{self.ds.name}/{self.modelo}/{self.instancia}"


def build_cases(bases=None, models=None, instances=50, seed=SEED):
    models = list(models or MODELS)
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise ValueError(f"Modelos desconhecidos: {unknown}; use {list(MODELS)}.")
    all_ds = datasets.load_all(seed)
    if bases:
        names = [ds.name for ds in all_ds]
        missing = [b for b in bases if b not in names]
        if missing:
            raise ValueError(f"Bases desconhecidas: {missing}; use {names}.")
    cases = []
    for ds in all_ds:
        if bases and ds.name not in bases:
            continue
        X_tr, X_te, y_tr, y_te, explain, background = split_and_select(ds, instances, seed)
        for name in models:
            model = MODELS[name]().fit(X_tr, y_tr)
            full = full_space(model.predict, ds.domains)
            data = sample_space(model.predict, X_tr)
            predicted = model.predict(explain).astype(int)
            phi = shap_for_predicted(model, background, explain, predicted)
            proba = model.predict_proba(explain)
            col = {int(k): j for j, k in enumerate(model.classes_)}
            acc = float(model.score(X_te, y_te))
            for k, v in enumerate(explain):
                c = int(predicted[k])
                f_full = Facts(full.waxp(v, c), ds.m)
                if not f_full.decision_depends_on_inputs:   
                    continue
                cases.append(Case(ds=ds, modelo=name, instancia=k, v=v.copy(), c=c,
                                  phi=np.asarray(phi[k], dtype=float).copy(),
                                  p_pred=float(proba[k, col[c]]),
                                  f_full=f_full, f_data=Facts(data.waxp(v, c), ds.m),
                                  full=full, data=data, acuracia=acc))
    return cases


def interleave(cases):
    base_order = {name: i for i, name in enumerate(datasets.ALL)}
    model_order = {name: i for i, name in enumerate(MODELS)}
    return sorted(cases, key=lambda cs: (cs.instancia, base_order[cs.ds.name],
                                         model_order[cs.modelo]))
