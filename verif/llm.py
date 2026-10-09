from __future__ import annotations

import difflib
import hashlib
import json
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from verif import claims as C
from verif.experiment import SEED

VERIFIED = ("SUF", "REL", "IRR", "CON")    
SHAP_ONLY = ("ORD", "DIR")                 
TYPES = VERIFIED + SHAP_ONLY + ("OUTRA",)
SPLIT = ("REL", "IRR", "DIR")              
TAU_IRR = 0.10                             
K_TOP = 3                               
EPS = 1e-9
NARRATION_TOKENS = 400
EXTRACTION_TOKENS = 1200

LEITURAS = {
    "SUF": "todos os atributos citados têm contribuição SHAP positiva (a favor da decisão)",
    "REL": "o atributo tem contribuição SHAP não nula",
    "IRR": ("a contribuição do atributo é desprezível (|phi_i| <= tau * max_j |phi_j|) ou é "
            "a de menor módulo entre os atributos fora dos três que mais favorecem a decisão"),
    "CON": "algum atributo citado tem contribuição SHAP positiva",
    "ORD": ("os atributos citados estão em ordem não crescente de |phi|; um atributo citado "
            "sozinho tem o maior |phi| ou o maior phi"),
    "DIR": "o sinal de phi_i coincide com o sentido afirmado (a favor: phi_i > 0; contra: phi_i < 0)",
}

def is_synthetic(ds):
    return ds.value_labels is None


def display_names(ds):
    if is_synthetic(ds):
        return [f"x{i + 1}" for i in range(ds.m)]
    return list(ds.feature_names)


def value_text(ds, i, value):
    if is_synthetic(ds):
        return f"valor {int(value)}"
    return f"faixa {ds.value_labels[i][int(value)]}"


def describe(ds, i, value):
    name = display_names(ds)[i]
    if is_synthetic(ds):
        return f"{name} = {int(value)}"
    return f"{name} na faixa {ds.value_labels[i][int(value)]}"


CONTEXT = {
    "breast-cancer": ("Um modelo de aprendizado de máquina classifica tumores de mama como "
                      "malignos ou benignos a partir de medidas dos núcleos celulares em imagens "
                      "de exame."),
    "wine": ("Um modelo de aprendizado de máquina identifica se um vinho é do cultivar 0 ou de "
             "outros cultivares a partir de análises químicas."),
    "iris": ("Um modelo de aprendizado de máquina identifica se uma flor de íris é da espécie "
             "versicolor ou de outras espécies a partir de medidas da flor."),
}


def _join(items, last=" e "):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + last + items[-1]


SUFFIXES = {"(média)": "(média) indica o valor médio na imagem",
            "(pior)": "(pior), a média dos três maiores valores",
            "(erro)": "(erro), o erro-padrão da medida"}


def context_text(ds):
    if is_synthetic(ds):
        return (f"Um modelo de aprendizado de máquina classifica casos descritos por {ds.m} "
                f"variáveis discretas, de x1 a x{ds.m}, nas classes \"{ds.class_names[0]}\" e "
                f"\"{ds.class_names[1]}\".")
    text = CONTEXT.get(ds.name, ds.description)
    used = [desc for suf, desc in SUFFIXES.items() if any(suf in n for n in ds.feature_names)]
    if used:
        text += f" Nos nomes das medidas, {_join(used)}."
    labels = []
    for lab in ds.value_labels:
        for item in lab:
            if item not in labels:
                labels.append(item)
    return (f"{text} Cada medida foi agrupada em faixas ({_join(labels, ' ou ')}) em relação "
            f"aos demais casos da base.")


def _num(x):
    return f"{x:+.3f}".replace(".", ",")


def _pct(p):
    return f"{100 * p:.0f}%"


SYSTEM_NARRATION = ("Você explica decisões de modelos de aprendizado de máquina para pessoas "
                    "sem formação técnica, em português do Brasil. Use apenas as informações "
                    "fornecidas.")

_FORMATO = ("Escreva em texto corrido, sem listas nem títulos, e não cite os valores "
            "numéricos das contribuições.")

INSTRUCOES = {
    "guiado": ("Escreva uma explicação curta, de 3 a 5 frases, dessa decisão para uma pessoa "
               "leiga. Na explicação, (1) diga quais fatores explicam a decisão; (2) diga qual "
               "fator mais pesou; (3) diga se algum fator não influenciou o resultado; e "
               "(4) diga o que poderia mudar a decisão. " + _FORMATO),
    "livre": ("Escreva uma explicação curta, de 3 a 5 frases, dessa decisão para uma pessoa "
              "leiga. " + _FORMATO),
}


def narration_messages(case, style="guiado"):
    ds, v, phi = case.ds, case.v, np.asarray(case.phi, dtype=float)
    names = display_names(ds)
    values = "\n".join(f"- {names[i]}: {value_text(ds, i, v[i])}" for i in range(ds.m))
    order = np.argsort(-np.abs(phi), kind="stable")
    contrib = "\n".join(f"- {names[i]}, {value_text(ds, i, v[i])}: {_num(phi[i])}"
                        for i in order)
    base = case.p_pred - float(phi.sum())
    user = (f"{context_text(ds)}\n\n"
            f"Valores do caso:\n{values}\n\n"
            f"Decisão do modelo: \"{case.label}\" (probabilidade estimada de "
            f"{_pct(case.p_pred)}).\n\n"
            f"Contribuições SHAP de cada atributo para essa decisão, da maior para a menor "
            f"em magnitude. Valores positivos favorecem \"{case.label}\", e negativos favorecem "
            f"\"{case.other_label}\". Partindo da probabilidade média de referência, de "
            f"{_pct(base)}, a soma das contribuições leva à probabilidade estimada.\n"
            f"{contrib}\n\n{INSTRUCOES[style]}")
    return [{"role": "system", "content": SYSTEM_NARRATION},
            {"role": "user", "content": user}]


SYSTEM_EXTRACTION = ("Você é um anotador cuidadoso. Sua tarefa é listar as afirmações que um "
                     "texto faz sobre a decisão de um modelo de classificação, sem acrescentar "
                     "nada que não esteja escrito.")

_EXEMPLO = {"afirmacoes": [
    {"trecho": "aprovado principalmente por causa da renda alta", "tipo": "SUF",
     "atributos": ["renda"], "atenuada": True, "sentido": ""},
    {"trecho": "principalmente por causa da renda alta", "tipo": "ORD",
     "atributos": ["renda"], "atenuada": True, "sentido": ""},
    {"trecho": "a renda alta, que pesou a favor", "tipo": "DIR",
     "atributos": ["renda"], "atenuada": False, "sentido": "a_favor"},
    {"trecho": "A idade não influenciou o resultado", "tipo": "IRR",
     "atributos": ["idade"], "atenuada": False, "sentido": ""},
    {"trecho": "Se as dívidas fossem altas, o resultado poderia mudar", "tipo": "CON",
     "atributos": ["dívidas"], "atenuada": False, "sentido": ""},
]}

GUIA_EXTRACAO = (
    "Liste cada afirmação que o texto faz sobre a decisão, com um destes tipos:\n"
    "- SUF: um ou mais atributos explicam ou determinam a decisão (\"a decisão se explica por "
    "A e B\", \"a decisão se deve a A e B\", \"A e B levaram à decisão\").\n"
    "- REL: um atributo influenciou ou contribuiu para a decisão (\"A influenciou\", \"A foi "
    "importante\", \"A contribuiu\").\n"
    "- IRR: um atributo não influenciou a decisão (\"B não influenciou\", \"B não teve "
    "efeito\", \"B não importou\").\n"
    "- CON: mudar um ou mais atributos poderia mudar a decisão (\"se A fosse diferente, a "
    "decisão poderia mudar\", \"com A baixo, o resultado seria outro\").\n"
    "- ORD: um atributo pesou mais que outros (\"A foi o fator que mais pesou\", \"A pesou "
    "mais que B\"); liste os atributos do mais para o menos importante.\n"
    "- DIR: um atributo favoreceu a decisão ou pesou contra ela (\"A favoreceu a decisão\", "
    "\"B pesou contra\"); indique o sentido.\n"
    "- OUTRA: qualquer outra afirmação sobre o modelo ou a decisão.\n"
    "Uma frase pode conter mais de uma afirmação. Não liste o simples anúncio da decisão "
    "(\"o modelo classificou o caso como X\"). Em \"atributos\", use só nomes da lista acima. "
    "Em \"atenuada\", use true quando a afirmação vier com ressalva (\"principalmente\", "
    "\"em parte\", \"provavelmente\", \"pode ter\") e false caso contrário. Em \"sentido\", "
    "use \"a_favor\" ou \"contra\" no tipo DIR e \"\" nos demais.\n\n"
    "Exemplo, com outro modelo (atributos: renda; idade; dívidas; decisão: \"aprovado\"):\n"
    "Texto: \"O pedido foi aprovado principalmente por causa da renda alta, que pesou a favor. "
    "A idade não influenciou o resultado. Se as dívidas fossem altas, o resultado poderia "
    "mudar.\"\n"
    "Resposta: " + json.dumps(_EXEMPLO, ensure_ascii=False) + "\n\n"
    "Responda apenas com o JSON, no formato do exemplo.")


def extraction_messages(case, text):
    names = display_names(case.ds)
    user = (f"Atributos do modelo (use exatamente estes nomes): {'; '.join(names)}.\n"
            f"Decisão do modelo: \"{case.label}\" (a outra classe é \"{case.other_label}\").\n\n"
            f"Texto:\n\"\"\"\n{text}\n\"\"\"\n\n{GUIA_EXTRACAO}")
    return [{"role": "system", "content": SYSTEM_EXTRACTION},
            {"role": "user", "content": user}]


def extraction_schema(names):
    item = {
        "type": "object",
        "properties": {
            "trecho": {"type": "string"},
            "tipo": {"type": "string", "enum": list(TYPES)},
            "atributos": {"type": "array", "items": {"type": "string", "enum": list(names)}},
            "atenuada": {"type": "boolean"},
            "sentido": {"type": "string", "enum": ["a_favor", "contra", ""]},
        },
        "required": ["trecho", "tipo", "atributos", "atenuada", "sentido"],
        "additionalProperties": False,
    }
    return {"type": "object",
            "properties": {"afirmacoes": {"type": "array", "items": item}},
            "required": ["afirmacoes"], "additionalProperties": False}


def clean_text(text):
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = t.strip().strip("\"“”").strip()
    return re.sub(r"[ \t]+", " ", t)

_STOP = {"o", "a", "os", "as", "de", "da", "do", "das", "dos", "e"}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(w for w in s.split() if w not in _STOP)


def map_attribute(name, names):
    name = re.split(r"\s+(?:na|em)\s+faixa\b|\s*=|,\s*(?:faixa|valor)\b|\s*\(faixa\b",
                    str(name))[0]
    n = _norm(name)
    if not n:
        return None
    normed = [_norm(x) for x in names]
    if n in normed:
        return normed.index(n)
    tokens = sorted(n.split())
    same = [j for j, x in enumerate(normed) if sorted(x.split()) == tokens]
    if len(same) == 1:
        return same[0]
    scores = sorted(((difflib.SequenceMatcher(None, n, x).ratio(), j)
                     for j, x in enumerate(normed)), reverse=True)
    if scores and scores[0][0] >= 0.85 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
        return scores[0][1]
    return None


def norm_type(t):
    u = _norm(t).replace(" ", "").upper()
    for prefix, kind in (("SUF", "SUF"), ("IRR", "IRR"), ("REL", "REL"), ("CON", "CON"),
                         ("ORD", "ORD"), ("DIR", "DIR"), ("SENT", "DIR")):
        if u.startswith(prefix):
            return kind
    return "OUTRA"


def _as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return _norm(x) in ("true", "sim", "verdadeiro", "1", "yes")


def _sentido(x):
    n = _norm(x).replace(" ", "_")
    if n.startswith("a_favor") or n in ("favor", "favoravel", "positivo"):
        return "a_favor"
    if n.startswith("contra") or n in ("desfavoravel", "negativo"):
        return "contra"
    return ""


def load_json(text):
    if not text:
        return None
    t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for j in range(start, len(t)):
            ch = t[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:j + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None


def parse_extraction(raw, names):
    obj = load_json(raw)
    if obj is None:
        return [], ["json_invalido"]
    items = obj.get("afirmacoes", obj.get("afirmações")) if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return [], ["sem_lista"]
    out, errors, seen = [], [], set()
    for it in items:
        if not isinstance(it, dict):
            errors.append("item_invalido")
            continue
        kind = norm_type(it.get("tipo", ""))
        attrs = it.get("atributos") or []
        if isinstance(attrs, str):
            attrs = [attrs]
        mapped = [(str(a), map_attribute(a, names)) for a in attrs]
        base = dict(tipo=kind, atenuada=_as_bool(it.get("atenuada", False)),
                    sentido=_sentido(it.get("sentido", "")) if kind == "DIR" else "",
                    trecho=str(it.get("trecho", "")).strip())
        if kind in SPLIT:
            groups = [([j], [] if j is not None else [a]) for a, j in mapped] or [([], [])]
            groups = [([j for j in g if j is not None], bad) for g, bad in groups]
        else:
            idx = []
            for _, j in mapped:
                if j is not None and j not in idx:
                    idx.append(j)
            groups = [(idx, [a for a, j in mapped if j is None])]
        for idx, bad in groups:
            if kind != "ORD":
                idx = sorted(set(idx))
            rec = dict(base, indices=tuple(int(i) for i in idx), nao_mapeados=tuple(bad))
            key = (kind, rec["indices"], rec["nao_mapeados"], rec["sentido"])
            if key in seen:
                continue
            seen.add(key)
            out.append(rec)
    return out, errors


def authorized(kind, idx, phi, sentido="", tau=TAU_IRR, k_top=K_TOP):
    phi = np.asarray(phi, dtype=float)
    a = np.abs(phi)
    idx = [int(i) for i in idx]
    if not idx:
        return None
    if kind == "SUF":
        return bool(all(phi[i] > EPS for i in idx))
    if kind == "REL":
        return bool(a[idx[0]] > EPS)
    if kind == "IRR":
        i = idx[0]
        if a[i] <= tau * a.max() + EPS:
            return True
        top = set(C.favorable(phi, min(k_top, len(phi) - 1)))
        rest = [j for j in range(len(phi)) if j not in top]
        return bool(i in rest and a[i] <= min(a[j] for j in rest) + EPS)
    if kind == "CON":
        return bool(any(phi[i] > EPS for i in idx))
    if kind == "ORD":
        if len(idx) == 1:
            return bool(a[idx[0]] >= a.max() - EPS or phi[idx[0]] >= phi.max() - EPS)
        return bool(all(a[idx[j]] >= a[idx[j + 1]] - EPS for j in range(len(idx) - 1)))
    if kind == "DIR":
        if sentido == "a_favor":
            return bool(phi[idx[0]] > EPS)
        if sentido == "contra":
            return bool(phi[idx[0]] < -EPS)
    return None


def assess(claim, case, tau=TAU_IRR):
    out = dict(claim)
    kind, idx = claim["tipo"], tuple(claim["indices"])
    names = display_names(case.ds)
    out["atributos"] = [names[i] for i in idx]
    mapped = len(idx) > 0 and not claim["nao_mapeados"]
    out["mapeada"] = mapped
    out["verificavel"] = bool(mapped and kind in VERIFIED)
    out["autorizada"] = (authorized(kind, idx, case.phi, claim.get("sentido", ""), tau)
                         if mapped and kind in VERIFIED + SHAP_ONLY else None)
    out.update(falsa=None, falsa_dados=None, irr_branda_falsa=None, reparo=None)
    if out["verificavel"]:
        cl = C.Claim(kind, tuple(sorted(idx)))
        out["falsa"] = not cl.holds(case.f_full)
        out["falsa_dados"] = not cl.holds(case.f_data)
        if kind == "IRR":
            out["irr_branda_falsa"] = case.f_full.contrastive(cl.mask)
        if kind == "SUF" and out["falsa"]:
            out["reparo"] = case.f_full.repair_distance(cl.mask)
    return out


TRECHOS_PADRAO = {"N1_suf_topk": "a decisão se explica por", "N2_rel_top1": "influenciou a decisão",
                  "N3_irr_ultimo": "não influenciou o resultado",
                  "N4_con_top1": "se seu valor fosse diferente, a decisão poderia mudar"}


def template_narrative(case, k=K_TOP):
    return C.shap_narrative(case.phi, k=min(k, case.ds.m - 1))


def template_claims(case, k=K_TOP):
   return [dict(tipo=cl.kind, indices=tuple(int(i) for i in cl.features), nao_mapeados=(),
                 atenuada=False, sentido="", trecho=TRECHOS_PADRAO[name])
            for name, cl in template_narrative(case, k).items()]


def verbalize_template(case, k=K_TOP):
    nar = template_narrative(case, k)
    names = display_names(case.ds)
    top = [describe(case.ds, i, case.v[i]) for i in nar["N1_suf_topk"].features]
    a = names[nar["N2_rel_top1"].features[0]]
    z = names[nar["N3_irr_ultimo"].features[0]]
    return (f"O modelo classificou o caso como {case.label}. A decisão se explica por "
            f"{_join(top)}. {a[:1].upper() + a[1:]} influenciou a decisão e, se seu valor "
            f"fosse diferente, a decisão poderia mudar. Já {z} não influenciou o resultado.")


def nearest_counterexample(case, idx):
    mask = C.to_mask(idx)
    for oracle, domain in ((case.data, "dados de treino"), (case.full, "espaço completo")):
        cex = oracle.counterexamples(case.v, case.c, mask)
        if len(cex):
            x = cex[int(np.argmin((cex != case.v).sum(axis=1)))]
            changed = [i for i in range(case.ds.m) if x[i] != case.v[i]]
            return dict(dominio=domain, classe=case.other_label,
                        mantidos=[describe(case.ds, i, case.v[i]) for i in idx],
                        alterados=[f"{describe(case.ds, i, x[i])} (em vez de "
                                   f"{value_text(case.ds, i, case.v[i])})" for i in changed])
    return None


def axp_witness(case, i):
    axps = [a for a in case.f_full.axps() if (a >> i) & 1]
    if not axps:
        return None
    best = min(axps, key=lambda a: (case.f_full.sizes[a], a))
    return [display_names(case.ds)[j] for j in C.to_features(best, case.ds.m)]

class LLMError(RuntimeError):
    pass


class HTTPStatusError(LLMError):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


@dataclass
class Reply:
    text: str
    meta: dict = field(default_factory=dict)


def _is_local(url):
    host = urllib.parse.urlparse(url).hostname or ""
    return host in ("localhost", "::1") or host.startswith("127.")


def http_json(url, payload=None, timeout=600, headers=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if data is None else "POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
              if _is_local(url) else urllib.request.build_opener())
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        raise HTTPStatusError(e.code, body) from None
    except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError) as e:
        raise LLMError(f"falha ao acessar {url}: {e}") from None
    except json.JSONDecodeError as e:
        raise LLMError(f"resposta inválida de {url}: {e}") from None


class OllamaClient:
    backend = "ollama"

    def __init__(self, model, host="http://localhost:11434", timeout=600, num_ctx=4096,
                 think=None, temperature=0.0, seed=SEED):
        self.model, self.host = model, host.rstrip("/")
        self.timeout, self.num_ctx, self.think = timeout, num_ctx, think
        self.temperature, self.seed = temperature, seed
        self.schema_ok = True      # vira False se o servidor recusar esquemas (versões antigas)
        self._info = None

    def info(self):
        if self._info is None:
            try:
                version = http_json(f"{self.host}/api/version", timeout=30).get("version")
                tags = http_json(f"{self.host}/api/tags", timeout=30).get("models", [])
            except LLMError as e:
                raise LLMError(f"Não consegui acessar o Ollama em {self.host}. Verifique se ele "
                               f"está aberto (no terminal: ollama serve). Detalhe: {e}") from None
            wanted = {self.model, f"{self.model}:latest"}
            entry = next((t for t in tags if t.get("name") in wanted or t.get("model") in wanted),
                         None)
            if entry is None:
                have = ", ".join(t.get("name", "?") for t in tags) or "nenhum"
                raise LLMError(f"O modelo '{self.model}' não está instalado no Ollama "
                               f"(instalados: {have}). Baixe-o com: ollama pull {self.model}")
            d = entry.get("details") or {}
            self._info = dict(backend=self.backend, servidor=f"Ollama {version}",
                              modelo=self.model, digest=entry.get("digest"),
                              familia=d.get("family"), parametros=d.get("parameter_size"),
                              quantizacao=d.get("quantization_level"))
        return self._info

    def signature(self):
        return dict(backend=self.backend, modelo=self.model, digest=self.info().get("digest"),
                    temperatura=self.temperature, semente=self.seed, num_ctx=self.num_ctx,
                    think=self.think)

    def chat(self, messages, schema=None, max_tokens=NARRATION_TOKENS):
        payload = dict(model=self.model, messages=messages, stream=False, keep_alive="10m",
                       options=dict(temperature=self.temperature, seed=self.seed,
                                    num_ctx=self.num_ctx, num_predict=max_tokens))
        if self.think is not None:
            payload["think"] = self.think
        if schema is not None:
            payload["format"] = schema if self.schema_ok else "json"
        t0 = time.time()
        try:
            resp = http_json(f"{self.host}/api/chat", payload, timeout=self.timeout)
        except HTTPStatusError as e:
            if schema is not None and self.schema_ok and e.status == 400:
                self.schema_ok = False      
                try:
                    return self.chat(messages, schema, max_tokens)
                except LLMError:
                    self.schema_ok = True
                    raise e from None
            raise
        msg = resp.get("message") or {}
        fmt = payload.get("format")
        meta = dict(segundos=round(time.time() - t0, 3),
                    tokens_entrada=resp.get("prompt_eval_count"),
                    tokens_saida=resp.get("eval_count"), fim=resp.get("done_reason"),
                    formato=None if fmt is None else ("esquema" if isinstance(fmt, dict) else fmt))
        return Reply(text=msg.get("content") or "", meta=meta)


class OpenAICompatClient:
    backend = "openai"

    def __init__(self, model, host="http://localhost:8080/v1", timeout=600, api_key=None,
                 temperature=0.0, seed=SEED):
        self.model, self.host = model, host.rstrip("/")
        self.timeout, self.api_key = timeout, api_key
        self.temperature, self.seed = temperature, seed
        self.schema_ok = True
        self._info = None

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def info(self):
        if self._info is None:
            try:
                data = http_json(f"{self.host}/models", timeout=30, headers=self._headers())
            except LLMError as e:
                raise LLMError(f"Não consegui acessar o servidor em {self.host}. Verifique o "
                               f"endereço (--host) e se o servidor está aberto. Detalhe: {e}") from None
            ids = [d.get("id") for d in data.get("data", []) if isinstance(d, dict)]
            if ids and self.model not in ids:
                raise LLMError(f"O modelo '{self.model}' não está disponível no servidor "
                               f"(disponíveis: {', '.join(map(str, ids[:20]))}).")
            self._info = dict(backend=self.backend, servidor="API compatível com a da OpenAI",
                              modelo=self.model)
        return self._info

    def signature(self):
        return dict(backend=self.backend, modelo=self.model, temperatura=self.temperature,
                    semente=self.seed)

    def chat(self, messages, schema=None, max_tokens=NARRATION_TOKENS):
        payload = dict(model=self.model, messages=messages, temperature=self.temperature,
                       seed=self.seed, max_tokens=max_tokens)
        if schema is not None and self.schema_ok:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "afirmacoes", "schema": schema, "strict": True}}
        t0 = time.time()
        try:
            resp = http_json(f"{self.host}/chat/completions", payload, timeout=self.timeout,
                             headers=self._headers())
        except HTTPStatusError as e:
            if schema is not None and self.schema_ok and e.status == 400:
                self.schema_ok = False       
                try:
                    return self.chat(messages, schema, max_tokens)
                except LLMError:
                    self.schema_ok = True
                    raise e from None
            raise
        choice = (resp.get("choices") or [{}])[0]
        usage = resp.get("usage") or {}
        meta = dict(segundos=round(time.time() - t0, 3),
                    tokens_entrada=usage.get("prompt_tokens"),
                    tokens_saida=usage.get("completion_tokens"),
                    fim=choice.get("finish_reason"),
                    formato=None if schema is None else ("esquema" if "response_format" in payload
                                                         else "instrução"))
        return Reply(text=(choice.get("message") or {}).get("content") or "", meta=meta)

class Cache:
    
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.data, self.hits, self.misses = {}, 0, 0
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        self.data[rec["chave"]] = rec["valor"]
                    except (json.JSONDecodeError, KeyError):
                        continue           

    @staticmethod
    def key(*parts):
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key):
        val = self.data.get(key)
        if val is None:
            self.misses += 1
        else:
            self.hits += 1
        return val

    def put(self, key, value):
        self.data[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"chave": key, "valor": value}, ensure_ascii=False) + "\n")


class LLMNarrator:
    
    def __init__(self, client, extractor=None, cache=None):
        self.client = client
        self.extractor = extractor or client
        self.cache = cache if cache is not None else Cache()
        self.new_calls = 0

    def info(self):
        info = dict(narrador=self.client.info())
        info["extrator"] = self.extractor.info() if self.extractor is not self.client else "o mesmo"
        return info

    def _ask(self, client, messages, schema, max_tokens):
        key = Cache.key(client.signature(), messages, schema, max_tokens)
        hit = self.cache.get(key)
        if hit is not None:
            return Reply(hit["text"], dict(hit.get("meta") or {}, cache=True))
        reply = client.chat(messages, schema=schema, max_tokens=max_tokens)
        self.new_calls += 1
        self.cache.put(key, {"text": reply.text, "meta": reply.meta})
        return reply

    def narrate(self, case, style="guiado"):
        r1 = self._ask(self.client, narration_messages(case, style), None, NARRATION_TOKENS)
        text = clean_text(r1.text)
        if not text:
            return dict(texto="", extracao_bruta="", meta=dict(narracao=r1.meta))
        schema = extraction_schema(display_names(case.ds))
        r2 = self._ask(self.extractor, extraction_messages(case, text), schema, EXTRACTION_TOKENS)
        return dict(texto=text, extracao_bruta=r2.text,
                    meta=dict(narracao=r1.meta, extracao=r2.meta))


class TemplateNarrator:
    new_calls = 0

    def info(self):
        return dict(narrador=dict(backend="mock", modelo="narrativa-padrao"), extrator="o mesmo")

    def narrate(self, case, style=None):
        names = display_names(case.ds)
        items = [dict(trecho=TRECHOS_PADRAO[name], tipo=cl.kind,
                      atributos=[names[i] for i in cl.features], atenuada=False, sentido="")
                 for name, cl in template_narrative(case).items()]
        return dict(texto=verbalize_template(case),
                    extracao_bruta=json.dumps({"afirmacoes": items}, ensure_ascii=False),
                    meta={})
