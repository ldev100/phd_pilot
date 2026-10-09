from dataclasses import dataclass

import numpy as np

KINDS = ("SUF", "CON", "REL", "IRR")


def popcounts(m):
    idx = np.arange(1 << m)
    sizes = np.zeros(1 << m, dtype=int)
    for i in range(m):
        sizes += (idx >> i) & 1
    return sizes


def to_mask(features):
    mask = 0
    for i in features:
        mask |= 1 << int(i)
    return mask


def to_features(mask, m):
    return tuple(i for i in range(m) if (mask >> i) & 1)


class Facts:
    
    def __init__(self, waxp, m):
        self.waxp = np.asarray(waxp, dtype=bool)
        self.m = m
        self.full = (1 << m) - 1
        self.sizes = popcounts(m)
        idx = np.arange(1 << m)
        self.wcxp = ~self.waxp[self.full ^ idx]       
        self._relevant = None
        self._axps = None
        self._cxps = None

    def sufficient(self, mask):
        return bool(self.waxp[mask])

    def contrastive(self, mask):
        return bool(self.wcxp[mask])

    @property
    def decision_depends_on_inputs(self):
        return not bool(self.waxp[0])

    def relevant(self):
        if self._relevant is None:
            idx = np.arange(1 << self.m)
            rel = np.zeros(self.m, dtype=bool)
            for i in range(self.m):
                bit = 1 << i
                S = idx[(idx & bit) == 0]
                rel[i] = bool(np.any(self.waxp[S | bit] & ~self.waxp[S]))
            self._relevant = rel
        return self._relevant

    @staticmethod
    def _minimal(table, m):
        out = []
        for S in np.flatnonzero(table):
            S = int(S)
            if all(not table[S & ~(1 << i)] for i in range(m) if (S >> i) & 1):
                out.append(S)
        return out

    def axps(self):
        if self._axps is None:
            self._axps = self._minimal(self.waxp, self.m)
        return self._axps

    def cxps(self):
        if self._cxps is None:
            self._cxps = self._minimal(self.wcxp, self.m)
        return self._cxps

    def _smallest(self, masks, weights):
        best = min(self.sizes[S] for S in masks)
        cands = [S for S in masks if self.sizes[S] == best]
        if weights is None:
            return cands[0]
        w = np.asarray(weights, dtype=float)
        return max(cands, key=lambda S: (w[list(to_features(S, self.m))].sum(), -S))

    def smallest_axp(self, weights=None):
       return self._smallest(self.axps(), weights)

    def smallest_cxp(self, weights=None):
        return self._smallest(self.cxps(), weights)

    def repair_distance(self, mask):
        cands = np.flatnonzero(self.waxp)
        cands = cands[(cands & mask) == mask]
        return int(self.sizes[cands].min() - self.sizes[mask])


@dataclass(frozen=True)
class Claim:
    kind: str            
    features: tuple      

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Tipo de afirmação desconhecido: {self.kind}")
        if self.kind in ("REL", "IRR") and len(self.features) != 1:
            raise ValueError("REL e IRR mencionam exatamente um atributo.")

    @property
    def mask(self):
        return to_mask(self.features)

    def holds(self, facts: Facts) -> bool:
        if self.kind == "SUF":
            return facts.sufficient(self.mask)
        if self.kind == "CON":
            return facts.contrastive(self.mask)
        rel = bool(facts.relevant()[self.features[0]])
        return rel if self.kind == "REL" else not rel

def shap_order(phi):
    return np.argsort(-np.abs(np.asarray(phi, dtype=float)), kind="stable")


def favorable(phi, k):
    phi = np.asarray(phi, dtype=float)
    order = np.argsort(-phi, kind="stable")
    pos = [int(i) for i in order if phi[i] > 0]
    return tuple(pos[:k]) if pos else (int(order[0]),)


def shap_narrative(phi, k=3):
   phi = np.asarray(phi, dtype=float)
    top = favorable(phi, k)
    rest = [int(i) for i in shap_order(phi) if int(i) not in top]
    z = rest[-1] if rest else int(shap_order(phi)[-1])
    return {
        "N1_suf_topk": Claim("SUF", top),
        "N2_rel_top1": Claim("REL", (top[0],)),
        "N3_irr_ultimo": Claim("IRR", (z,)),
        "N4_con_top1": Claim("CON", (top[0],)),
    }


def verified_narrative(facts: Facts, phi):
    phi = np.asarray(phi, dtype=float)
    w = np.maximum(phi, 0.0) + 1e-12 * np.abs(phi)   # contribuições a favor; |phi| desempata
    rel = facts.relevant()
    rel_idx = np.flatnonzero(rel)
    claims = {
        "V1_suf": Claim("SUF", to_features(facts.smallest_axp(w), facts.m)),
        "V2_rel": Claim("REL", (int(rel_idx[np.argmax(phi[rel_idx])]),)),
        "V4_con": Claim("CON", to_features(facts.smallest_cxp(w), facts.m)),
    }
    if (~rel).any():
        irr = np.flatnonzero(~rel)
        claims["V3_irr"] = Claim("IRR", (int(irr[np.argmin(np.abs(phi[irr]))]),))
    for name, claim in claims.items():
        if not claim.holds(facts):
            raise AssertionError(f"Afirmação verificada {name} deveria valer por construção.")
    return claims

def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " e " + items[-1]


def _cap(text):
    return text[:1].upper() + text[1:]


def verbalize_shap(claims, ds, v, label):
    d = lambda i: ds.describe_value(i, v[i])
    top = [d(i) for i in claims["N1_suf_topk"].features]
    a = ds.feature_names[claims["N2_rel_top1"].features[0]]
    z = ds.feature_names[claims["N3_irr_ultimo"].features[0]]
    return (f"O modelo classificou o caso como {label}. A decisão se explica por "
            f"{_join(top)}. {_cap(a)} influenciou a decisão e, se seu valor fosse diferente, "
            f"a decisão poderia mudar. Já {z} não influenciou o resultado.")


def verbalize_verified(claims, ds, v, label):
    d = lambda i: ds.describe_value(i, v[i])
    suf = [d(i) for i in claims["V1_suf"].features]
    con = [ds.feature_names[i] for i in claims["V4_con"].features]
    text = (f"O modelo classificou o caso como {label}. Estes valores bastam para essa "
            f"decisão: {_join(suf)}; qualquer caso com eles recebe a mesma classe. ")
    if len(con) == 1:
        text += f"Mudar apenas {con[0]} poderia reverter a decisão."
    else:
        n = {2: "dois", 3: "três", 4: "quatro", 5: "cinco"}.get(len(con), str(len(con)))
        text += (f"Nenhuma mudança em um único atributo reverte a decisão; seria preciso "
                 f"alterar ao menos {n} atributos ao mesmo tempo, por exemplo {_join(con)}.")
    if "V3_irr" in claims:
        text += f" {_cap(ds.feature_names[claims['V3_irr'].features[0]])} não influenciou a decisão."
    return text
