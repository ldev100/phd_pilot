import argparse
import datetime as dt
import glob
import json
import os
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from verif import llm
from verif.experiment import MODELS, REAL, SEED, build_cases, interleave

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

GROUPS = ("AD", "RF", "GB", "todos")
NAMES_TEX = {"AD": "Árvore de decisão", "RF": r"\textit{Random forest}",
             "GB": r"\textit{Boosting}", "todos": "Todos os modelos"}
TYPE_NAMES = {"SUF": "Suficiência", "REL": "Relevância", "IRR": "Irrelevância",
              "CON": "Contraste", "ORD": "Ordem", "DIR": "Sentido", "OUTRA": "Outra"}
FAIL_ERRORS = {"json_invalido", "sem_lista", "narrativa_vazia", "extracao_vazia"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["ollama", "openai", "mock"], default="ollama",
                   help="servidor do LLM: ollama (padrão), openai (API compatível com a da "
                        "OpenAI: llama.cpp, vLLM, LM Studio) ou mock (sem LLM)")
    p.add_argument("--model", help="nome do modelo no servidor (ex.: qwen2.5:7b)")
    p.add_argument("--extractor-model",
                   help="modelo que extrai as afirmações (padrão: o mesmo do narrador)")
    p.add_argument("--host", help="endereço do servidor (padrão: http://localhost:11434 no "
                                  "Ollama e http://localhost:8080/v1 na API da OpenAI)")
    p.add_argument("--style", choices=["guiado", "livre"], default="guiado",
                   help="instrução de narração: guiado (pede fatores, o principal, algum sem "
                        "influência e o que mudaria a decisão) ou livre (padrão: guiado)")
    p.add_argument("--bases", nargs="+", help="bases a usar (padrão: todas as nove)")
    p.add_argument("--models", nargs="+", choices=list(MODELS),
                   help="modelos de AM (padrão: AD RF GB)")
    p.add_argument("--instances", type=int, default=50,
                   help="instâncias explicadas por base e modelo (padrão: 50, como no piloto)")
    p.add_argument("--max-cases", type=int,
                   help="limita o número de narrativas, em ordem intercalada entre bases e modelos")
    p.add_argument("--timeout", type=float, default=600,
                   help="tempo limite por chamada ao LLM, em segundos (padrão: 600)")
    p.add_argument("--num-ctx", type=int, default=4096, help="contexto do Ollama (padrão: 4096)")
    p.add_argument("--think", choices=["auto", "on", "off", "low", "medium", "high"],
                   default="auto",
                   help="raciocínio de modelos que pensam antes de responder, no Ollama: off "
                        "desliga (ex.: qwen3); low/medium/high para gpt-oss (padrão: auto, "
                        "não envia o parâmetro)")
    p.add_argument("--tau", type=float, default=llm.TAU_IRR,
                   help="fração da maior contribuição abaixo da qual ela é desprezível "
                        "(leitura-padrão da irrelevância; padrão: 0.10)")
    p.add_argument("--margin", type=float, default=10.0,
                   help="margem de não inferioridade, em pontos percentuais (padrão: 10)")
    p.add_argument("--annotation-sample", type=int, default=40,
                   help="narrativas sorteadas para conferência manual da extração (padrão: 40)")
    p.add_argument("--out", default="results_llm", help="pasta de saída (padrão: results_llm)")
    p.add_argument("--quick", action="store_true",
                   help="execução reduzida: 2 instâncias por base e modelo")
    p.add_argument("--combine", nargs="+", metavar="PASTA",
                   help="não gera narrativas: reúne os resultados de execuções anteriores")
    args = p.parse_args(argv)
    if args.quick:
        args.instances = min(args.instances, 2)
    if not args.combine and args.backend != "mock" and not args.model:
        p.error("informe o modelo (ex.: --model qwen2.5:7b) ou use --backend mock")
    return args


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "modelo"


def run_dir(args):
    if args.backend == "mock":
        return Path(args.out) / "mock"
    name = slug(args.model)
    if args.extractor_model and args.extractor_model != args.model:
        name += f"__extrator_{slug(args.extractor_model)}"
    return Path(args.out) / name / args.style


def make_narrator(args, cache):
    if args.backend == "mock":
        return llm.TemplateNarrator()
    think = {"auto": None, "on": True, "off": False}.get(args.think, args.think)
    if args.backend == "ollama":
        host = args.host or "http://localhost:11434"
        make = lambda m: llm.OllamaClient(m, host=host, timeout=args.timeout,
                                          num_ctx=args.num_ctx, think=think)
    else:
        host = args.host or "http://localhost:8080/v1"
        key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        make = lambda m: llm.OpenAICompatClient(m, host=host, timeout=args.timeout, api_key=key)
    narrator = make(args.model)
    same = not args.extractor_model or args.extractor_model == args.model
    return llm.LLMNarrator(narrator, narrator if same else make(args.extractor_model), cache)

def process(case, narrator, args):
    names = llm.display_names(case.ds)
    out = narrator.narrate(case, args.style)
    if not out["texto"]:
        claims, errors = [], ["narrativa_vazia"]
    elif not (out["extracao_bruta"] or "").strip():
        claims, errors = [], ["extracao_vazia"]
    else:
        claims, errors = llm.parse_extraction(out["extracao_bruta"], names)
    return dict(case=case, texto=out["texto"], extracao_bruta=out["extracao_bruta"],
                erros=errors, falha=bool(FAIL_ERRORS & set(errors)), meta=out["meta"],
                llm=[llm.assess(c, case, args.tau) for c in claims],
                padrao=[llm.assess(c, case, args.tau) for c in llm.template_claims(case)])


def fmt_duration(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {(seconds % 3600) // 60:02d} min"


def run(args, narrator, cases):
    records, t0, interrupted, error = [], time.time(), False, None
    every = 1 if args.backend != "mock" else 200
    timed, timed_s = 0, 0.0
    try:
        for n, case in enumerate(cases, 1):
            calls_before, t_case = narrator.new_calls, time.time()
            rec = process(case, narrator, args)
            records.append(rec)
            if narrator.new_calls > calls_before:
                timed += 1
                timed_s += time.time() - t_case
            if n % every == 0 or n == len(cases):
                ver = [c for c in rec["llm"] if c["verificavel"]]
                eta = (timed_s / timed) * (len(cases) - n) if timed else 0.0
                status = "falha na extração" if rec["falha"] else (
                    f"{len(rec['llm'])} afirmações, {len(ver)} verificáveis, "
                    f"{sum(bool(c['falsa']) for c in ver)} falsas")
                print(f"[{n:>4d}/{len(cases)}] {case.key:<24s} {status} | "
                      f"faltam ~{fmt_duration(eta)}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrompido. Salvando os resultados parciais; rode o mesmo comando para "
              "continuar (as respostas já obtidas estão no cache).", flush=True)
    except llm.LLMError as e:
        interrupted, error = True, str(e)
        print(f"\nErro na chamada ao LLM: {e}\nSalvando os resultados parciais; corrija o "
              f"problema e rode o mesmo comando para continuar.", flush=True)
    return records, time.time() - t0, interrupted, error

def group_of(base):
    return "reais" if base in REAL else "sinteticas"


def claim_table(records):
    rows = []
    for r in records:
        cs = r["case"]
        for origin in ("llm", "padrao"):
            for c in r[origin]:
                rows.append(dict(
                    base=cs.ds.name, grupo=group_of(cs.ds.name), modelo=cs.modelo,
                    instancia=cs.instancia, narrativa=origin, falha_extracao=r["falha"],
                    tipo=c["tipo"], atributos="; ".join(c["atributos"]),
                    nao_mapeados="; ".join(c["nao_mapeados"]), mapeada=c["mapeada"],
                    verificavel=c["verificavel"], autorizada=c["autorizada"],
                    falsa=c["falsa"], falsa_dados=c["falsa_dados"],
                    irr_branda_falsa=c["irr_branda_falsa"], reparo=c["reparo"],
                    atenuada=c["atenuada"], sentido=c["sentido"], trecho=c["trecho"]))
    cols = ["base", "grupo", "modelo", "instancia", "narrativa", "falha_extracao", "tipo",
            "atributos", "nao_mapeados", "mapeada", "verificavel", "autorizada", "falsa",
            "falsa_dados", "irr_branda_falsa", "reparo", "atenuada", "sentido", "trecho"]
    return pd.DataFrame(rows, columns=cols)


def narrative_table(records):
    rows = []
    for r in records:
        cs, L, P = r["case"], r["llm"], r["padrao"]
        ver = [c for c in L if c["verificavel"]]
        meta = r["meta"] or {}
        seconds = sum((meta.get(k) or {}).get("segundos") or 0 for k in ("narracao", "extracao"))
        tokens = sum((meta.get(k) or {}).get("tokens_saida") or 0 for k in ("narracao", "extracao"))
        cached = all((meta.get(k) or {}).get("cache", False) for k in ("narracao", "extracao")
                     if k in meta) if meta else False
        row = dict(base=cs.ds.name, grupo=group_of(cs.ds.name), modelo=cs.modelo,
                   instancia=cs.instancia, classe=cs.label, m=cs.ds.m,
                   falha_extracao=r["falha"], erros=";".join(r["erros"]),
                   palavras=len(r["texto"].split()), n_afirmacoes=len(L),
                   n_verificaveis=len(ver), n_nao_mapeadas=sum(not c["mapeada"] for c in L
                                                               if c["tipo"] != "OUTRA"),
                   alguma_falsa=any(c["falsa"] for c in ver),
                   alguma_falsa_dados=any(c["falsa_dados"] for c in ver),
                   herdado=any(c["falsa"] and c["autorizada"] is True for c in ver),
                   narrador=any(c["falsa"] and c["autorizada"] is False for c in ver),
                   padrao_alguma_falsa=any(c["falsa"] for c in P),
                   padrao_alguma_falsa_dados=any(c["falsa_dados"] for c in P),
                   segundos_llm=round(seconds, 2), tokens_saida=tokens, do_cache=cached)
        for t in llm.VERIFIED:
            row[f"tem_{t}"] = any(c["tipo"] == t for c in ver)
            row[f"{t}_falsa"] = any(c["tipo"] == t and c["falsa"] for c in ver)
        rows.append(row)
    return pd.DataFrame(rows)


def _pct(series):
    s = pd.Series(series, dtype="float64").dropna()
    return float(s.mean() * 100) if len(s) else float("nan")


def _mask(df, group):
    return np.ones(len(df), bool) if group == "todos" else (df["modelo"] == group).values


def by_model(narr, claims):
    ok = narr[~narr.falha_extracao]
    ok_keys = set(zip(ok.base, ok.modelo, ok.instancia))
    keep = np.array([k in ok_keys for k in zip(claims.base, claims.modelo, claims.instancia)],
                    dtype=bool) if len(claims) else np.zeros(0, bool)
    cl = claims[keep]
    rows = []
    for g in GROUPS:
        n = ok[_mask(ok, g)]
        c = cl[_mask(cl, g)] if len(cl) else cl
        L = c[(c.narrativa == "llm") & c.verificavel.astype(bool)] if len(c) else c
        P = c[c.narrativa == "padrao"] if len(c) else c
        row = dict(modelo=g, narrativas=len(n),
                   falhas_extracao=int((narr.falha_extracao & _mask(narr, g)).sum()),
                   palavras=n.palavras.mean() if len(n) else np.nan,
                   afirmacoes_por_narrativa=n.n_afirmacoes.mean() if len(n) else np.nan,
                   verificaveis_por_narrativa=n.n_verificaveis.mean() if len(n) else np.nan,
                   com_verificavel=_pct(n.n_verificaveis > 0),
                   alguma_falsa=_pct(n.alguma_falsa), padrao_alguma_falsa=_pct(n.padrao_alguma_falsa),
                   alguma_falsa_dados=_pct(n.alguma_falsa_dados),
                   padrao_alguma_falsa_dados=_pct(n.padrao_alguma_falsa_dados),
                   herdado=_pct(n.herdado), narrador=_pct(n.narrador))
        false_claims = L[L.falsa.astype(bool)] if len(L) else L
        row["falsas_autorizadas"] = _pct(false_claims.autorizada == True) if len(false_claims) else np.nan
        for t in llm.VERIFIED:
            lt = L[L.tipo == t] if len(L) else L
            pt = P[P.tipo == t] if len(P) else P
            row[f"{t}_n"] = len(lt)
            row[f"{t}_falsa"] = _pct(lt.falsa) if len(lt) else np.nan
            row[f"{t}_falsa_dados"] = _pct(lt.falsa_dados) if len(lt) else np.nan
            row[f"{t}_autorizada"] = _pct(lt.autorizada) if len(lt) else np.nan
            row[f"padrao_{t}_falsa"] = _pct(pt.falsa) if len(pt) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def by_type(claims, narr):
    """Cobertura da linguagem, fidelidade ao SHAP, atenuação e leitura branda, por tipo."""
    ok_keys = set(zip(narr.base[~narr.falha_extracao], narr.modelo[~narr.falha_extracao],
                      narr.instancia[~narr.falha_extracao]))
    L = claims[(claims.narrativa == "llm")]
    L = L[[k in ok_keys for k in zip(L.base, L.modelo, L.instancia)]] if len(L) else L
    total = len(L)
    rows = []
    for t in llm.TYPES:
        lt = L[L.tipo == t]
        v = lt[lt.verificavel.astype(bool)] if len(lt) else lt
        row = dict(tipo=t, afirmacoes=len(lt), percentual=100 * len(lt) / total if total else np.nan,
                   mapeadas=_pct(lt.mapeada) if len(lt) else np.nan,
                   autorizadas=_pct(lt.autorizada.dropna()) if len(lt) else np.nan,
                   falsas=_pct(v.falsa) if len(v) else np.nan,
                   falsas_dados=_pct(v.falsa_dados) if len(v) else np.nan,
                   atenuadas=_pct(lt.atenuada) if len(lt) else np.nan)
        for flag, name in ((True, "atenuadas"), (False, "nao_atenuadas")):
            sub = v[v.atenuada == flag] if len(v) else v
            row[f"n_{name}"] = len(sub)
            row[f"falsas_{name}"] = _pct(sub.falsa) if len(sub) else np.nan
        if t == "IRR":
            row["falsas_leitura_branda"] = _pct(v.irr_branda_falsa) if len(v) else np.nan
        if t == "SUF" and len(v):
            rep = v.reparo.dropna()
            row["reparo_medio"] = float(rep.mean()) if len(rep) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def by_base(narr):
    ok = narr[~narr.falha_extracao]
    if not len(ok):
        return pd.DataFrame(columns=["base", "modelo", "narrativas", "alguma_falsa",
                                     "padrao_alguma_falsa", "diferenca"])
    g = ok.groupby(["base", "modelo"], sort=False)
    out = g.agg(narrativas=("alguma_falsa", "size"),
                alguma_falsa=("alguma_falsa", "mean"),
                padrao_alguma_falsa=("padrao_alguma_falsa", "mean"),
                herdado=("herdado", "mean"), narrador=("narrador", "mean")).reset_index()
    for col in ("alguma_falsa", "padrao_alguma_falsa", "herdado", "narrador"):
        out[col] *= 100
    out["diferenca"] = out.alguma_falsa - out.padrao_alguma_falsa
    return out


def noninferiority(narr, margin, n_boot=10000, seed=SEED):
    from scipy.stats import wilcoxon
    ok = narr[~narr.falha_extracao]
    rows = []
    for g in GROUPS:
        n = ok[_mask(ok, g)]
        if not len(n):
            continue
        per = n.groupby("base", sort=False).agg(llm=("alguma_falsa", "mean"),
                                                padrao=("padrao_alguma_falsa", "mean"))
        d = (per.llm - per.padrao).to_numpy() * 100
        k = len(d)
        rng = np.random.default_rng(seed)
        boots = d[rng.integers(0, k, size=(n_boot, k))].mean(axis=1)
        lb = float(np.percentile(boots, 5))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        shifted = d + margin
        try:
            p = float(wilcoxon(shifted, alternative="greater").pvalue) if np.any(shifted != 0) and k >= 2 else np.nan
        except ValueError:
            p = np.nan
        rows.append(dict(modelo=g, bases=k, llm=float(per.llm.mean() * 100),
                         padrao=float(per.padrao.mean() * 100), diferenca_media=float(d.mean()),
                         limite_inferior_95=lb, ic95_inf=float(lo), ic95_sup=float(hi),
                         margem=margin, nao_inferior=bool(lb > -margin), wilcoxon_p=p,
                         indicativo=k < 5))
    return pd.DataFrame(rows)


def annotation_sheet(records, n, seed=SEED):
    pool = {}
    for r in records:
        if r["texto"]:
            pool.setdefault(r["case"].ds.name, []).append(r)
    rng = np.random.default_rng(seed)
    for name in pool:
        order = rng.permutation(len(pool[name]))
        pool[name] = [pool[name][i] for i in order]
    chosen, i = [], 0
    target = min(n, sum(len(v) for v in pool.values()))
    while len(chosen) < target:
        for name in pool:
            if i < len(pool[name]) and len(chosen) < target:
                chosen.append(pool[name][i])
        i += 1
    rows = []
    blank = dict(anotador1_correta="", anotador1_tipo_correto="", anotador2_correta="",
                 anotador2_tipo_correto="", observacoes="")
    for j, r in enumerate(chosen, 1):
        cs = r["case"]
        head = dict(narrativa_id=j, base=cs.ds.name, modelo=cs.modelo, instancia=cs.instancia,
                    narrativa=r["texto"])
        for c in r["llm"]:
            rows.append(dict(head, trecho=c["trecho"], tipo_extraido=c["tipo"],
                             atributos_extraidos="; ".join(c["atributos"] or c["nao_mapeados"]),
                             atenuada="sim" if c["atenuada"] else "não", **blank))
        rows.append(dict(head, trecho="", tipo_extraido="FALTANTE",
                         atributos_extraidos="(liste afirmações do texto que não foram extraídas)",
                         atenuada="", **blank))
    return pd.DataFrame(rows)

def _jsonable(x):
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (np.ndarray, tuple)):
        return list(x)
    raise TypeError(f"tipo não serializável: {type(x)}")


def full_record(r):
    cs = r["case"]
    names = llm.display_names(cs.ds)
    claims = []
    for c in r["llm"]:
        c = dict(c)
        if c["tipo"] == "SUF" and c["falsa"]:
            c["contraexemplo"] = llm.nearest_counterexample(cs, c["indices"])
        if c["tipo"] == "IRR" and c["falsa"]:
            c["testemunha_axp"] = llm.axp_witness(cs, c["indices"][0])
        claims.append(c)
    return dict(base=cs.ds.name, modelo=cs.modelo, instancia=cs.instancia, classe=cs.label,
                probabilidade=round(cs.p_pred, 4),
                valores={names[i]: llm.value_text(cs.ds, i, cs.v[i]) for i in range(cs.ds.m)},
                shap={names[i]: round(float(cs.phi[i]), 4)
                      for i in np.argsort(-np.abs(cs.phi), kind="stable")},
                narrativa=r["texto"], extracao_bruta=r["extracao_bruta"], erros=r["erros"],
                afirmacoes=claims, narrativa_padrao=llm.verbalize_template(cs),
                afirmacoes_padrao=r["padrao"], meta=r["meta"])


def pick_examples(records, k=3):
    order = {name: i for i, name in enumerate(REAL)}
    cands = []
    for r in records:
        ver = [c for c in r["llm"] if c["verificavel"]]
        if r["falha"] or not any(c["falsa"] for c in ver):
            continue
        both = any(c["falsa"] and c["autorizada"] for c in ver) and \
            any(c["falsa"] and c["autorizada"] is False for c in ver)
        cs = r["case"]
        cands.append(((cs.ds.name not in REAL, not both, order.get(cs.ds.name, 9),
                       cs.modelo, cs.instancia), r))
    cands.sort(key=lambda t: t[0])
    chosen, seen = [], set()
    for _, r in cands:
        key = (r["case"].ds.name, r["case"].modelo)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(r)
        if len(chosen) == k:
            break
    return chosen


def describe_claim(c):
    attrs = ", ".join(c["atributos"]) if c["atributos"] else ", ".join(c["nao_mapeados"])
    if c["verificavel"]:
        verdict = "**falsa**" if c["falsa"] else "verdadeira"
        verdict += "; nos dados, " + ("falsa" if c["falsa_dados"] else "verdadeira")
    elif c["tipo"] in llm.VERIFIED and not c["mapeada"]:
        verdict = "não verificada (atributo não reconhecido)"
    else:
        verdict = "não verificada (fora da linguagem)"
    auth = {True: "autorizada pelo SHAP", False: "não autorizada pelo SHAP", None: ""}[c["autorizada"]]
    extra = " (atenuada)" if c["atenuada"] else ""
    quote = f" \"{c['trecho']}\"" if c["trecho"] else ""
    return f"{c['tipo']}({attrs}){extra}{quote}: {verdict}" + (f"; {auth}" if auth else "")

def fmt(x, nd=0):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def br(x, nd=1):
    return fmt(x, nd).replace(".", ",")


def write_latex(path, bm, run_info):
    lines = [r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{4}{c}{Afirmações falsas do LLM, por tipo (\%)} & "
             r"\multicolumn{2}{c}{Alguma falsa (\%)} & \multicolumn{2}{c}{Origem (\%)}\\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
             r"Modelo & Narrativas & Suf. & Rel. & Irr. & Con. & LLM & Padrão & Herdado & Narrador\\",
             r"\midrule"]
    for g in GROUPS:
        r = bm[bm.modelo == g]
        if not len(r):
            continue
        r = r.iloc[0]
        if g == "todos":
            lines.append(r"\midrule")
        lines.append(f"{NAMES_TEX[g]} & {int(r.narrativas)} & {fmt(r.SUF_falsa)} & "
                     f"{fmt(r.REL_falsa)} & {fmt(r.IRR_falsa)} & {fmt(r.CON_falsa)} & "
                     f"{fmt(r.alguma_falsa)} & {fmt(r.padrao_alguma_falsa)} & "
                     f"{fmt(r.herdado)} & {fmt(r.narrador)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    note = (f"% Narrador: {run_info['narrador']}; estilo: {run_info['estilo']}. "
            "Herdado/Narrador: % de narrativas com afirmação falsa autorizada/não autorizada "
            "pela saída do SHAP.")
    path.write_text(note + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path, bm, bt, bb, ni, examples, run_info, args):
    L = ["# Narrativas redigidas por LLM: resultados", "",
         f"Narrador: {run_info['narrador']}; extrator: {run_info['extrator']}; "
         f"estilo de instrução: {run_info['estilo']}.",
         f"Narrativas: {run_info['narrativas']} de {run_info['casos_previstos']} previstas"
         + (" (execução interrompida)" if run_info["interrompido"] else "")
         + f"; falhas de extração: {run_info['falhas_extracao']}.",
         "Vereditos no espaço de entrada completo, salvo indicação. A narrativa-padrão é a do "
         "piloto, avaliada nas mesmas instâncias.", "",
         "## Afirmações falsas: LLM e narrativa-padrão", "",
         "| Modelo | Narrativas | Afirmações/narr. | Verificáveis/narr. | Suf. | Rel. | Irr. | "
         "Con. | Alguma (LLM) | Alguma (padrão) | Alguma nos dados (LLM / padrão) | "
         "Herdado | Narrador |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for _, r in bm.iterrows():
        L.append(f"| {r.modelo} | {int(r.narrativas)} | {fmt(r.afirmacoes_por_narrativa, 1)} | "
                 f"{fmt(r.verificaveis_por_narrativa, 1)} | {fmt(r.SUF_falsa)} | "
                 f"{fmt(r.REL_falsa)} | {fmt(r.IRR_falsa)} | {fmt(r.CON_falsa)} | "
                 f"{fmt(r.alguma_falsa)} | {fmt(r.padrao_alguma_falsa)} | "
                 f"{fmt(r.alguma_falsa_dados)} / {fmt(r.padrao_alguma_falsa_dados)} | "
                 f"{fmt(r.herdado)} | {fmt(r.narrador)} |")
    L += ["", "Suf., Rel., Irr. e Con.: % das afirmações do LLM de cada tipo que são falsas. "
          "Alguma: % das narrativas com ao menos uma afirmação verificável falsa. Herdado e "
          "Narrador: % das narrativas com afirmação falsa autorizada e não autorizada pela "
          "saída do SHAP, respectivamente (uma narrativa pode ter as duas).", "",
          "Mesmos tipos de afirmação na narrativa-padrão (% falsas): "
          + "; ".join(f"{r.modelo}: Suf. {fmt(r.padrao_SUF_falsa)}, Rel. {fmt(r.padrao_REL_falsa)}, "
                      f"Irr. {fmt(r.padrao_IRR_falsa)}, Con. {fmt(r.padrao_CON_falsa)}"
                      for _, r in bm.iterrows()) + ".", ""]
    tot = bm[bm.modelo == "todos"]
    if len(tot) and not pd.isna(tot.iloc[0].falsas_autorizadas):
        share = tot.iloc[0].falsas_autorizadas
        L += [f"Das afirmações verificáveis falsas do LLM, {fmt(share)}% eram autorizadas pela "
              f"saída do SHAP (erro herdado da fonte) e {fmt(100 - share)}% não eram (erro do "
              f"narrador).", ""]
    L += ["## Não inferioridade pareada por base (H1)", "",
          f"Margem fixada antes da execução: {br(args.margin)} pontos percentuais. "
          "Não inferior: o limite inferior unilateral de 95% da diferença média (LLM menos "
          "padrão), por reamostragem das bases, fica acima de menos a margem.", "",
          "| Modelo | Bases | LLM (%) | Padrão (%) | Diferença média (p.p.) | Limite inferior 95% | "
          "IC 95% | Não inferior? | Wilcoxon p |",
          "|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for _, r in ni.iterrows():
        flag = ("sim" if r.nao_inferior else "não") + (" (indicativo: menos de 5 bases)" if r.indicativo else "")
        L.append(f"| {r.modelo} | {int(r.bases)} | {fmt(r.llm)} | {fmt(r.padrao)} | "
                 f"{fmt(r.diferenca_media, 1)} | {fmt(r.limite_inferior_95, 1)} | "
                 f"{fmt(r.ic95_inf, 1)} a {fmt(r.ic95_sup, 1)} | {flag} | {fmt(r.wilcoxon_p, 3)} |")
    L += ["", "## Cobertura da linguagem e fidelidade à saída do SHAP", "",
          "| Tipo | Afirmações | % do total | Atributos reconhecidos (%) | Autorizadas pelo SHAP (%) "
          "| Falsas (%) | Falsas nos dados (%) | Atenuadas (%) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in bt.iterrows():
        L.append(f"| {r.tipo} | {int(r.afirmacoes)} | {fmt(r.percentual)} | {fmt(r.mapeadas)} | "
                 f"{fmt(r.autorizadas)} | {fmt(r.falsas)} | {fmt(r.falsas_dados)} | "
                 f"{fmt(r.atenuadas)} |")
    L += ["", "ORD e DIR são conferidas só contra a saída do SHAP (fidelidade, como na "
          "literatura); OUTRA fica fora da linguagem. A cobertura indica quais extensões da "
          "linguagem (necessidade, contrafactuais com valores, ordem) mais fariam falta.", "",
          "## Afirmações atenuadas e leitura branda da irrelevância", "",
          "| Tipo | Atenuadas (n) | Falsas entre atenuadas (%) | Não atenuadas (n) | "
          "Falsas entre não atenuadas (%) |", "|---|---:|---:|---:|---:|"]
    for _, r in bt[bt.tipo.isin(llm.VERIFIED)].iterrows():
        L.append(f"| {r.tipo} | {int(r.n_atenuadas)} | {fmt(r.falsas_atenuadas)} | "
                 f"{int(r.n_nao_atenuadas)} | {fmt(r.falsas_nao_atenuadas)} |")
    irr = bt[bt.tipo == "IRR"].iloc[0]
    L += ["", f"Irrelevância lida de forma estrita (o atributo não pertence a nenhuma explicação "
          f"abdutiva): {fmt(irr.falsas)}% falsas. Lida de forma branda (mudar só esse atributo "
          f"não reverteria a decisão): {fmt(irr.get('falsas_leitura_branda', np.nan))}% falsas.", ""]
    suf = bt[bt.tipo == "SUF"].iloc[0]
    if "reparo_medio" in suf and not pd.isna(suf.get("reparo_medio", np.nan)):
        L += [f"Suficiências falsas do LLM são reparadas acrescentando, em média, "
              f"{br(suf.reparo_medio, 2)} atributos.", ""]
    L += ["## Por base", "",
          "| Base | Modelo | Narrativas | Alguma (LLM) | Alguma (padrão) | Diferença | Herdado | Narrador |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in bb.iterrows():
        L.append(f"| {r.base} | {r.modelo} | {int(r.narrativas)} | {fmt(r.alguma_falsa)} | "
                 f"{fmt(r.padrao_alguma_falsa)} | {fmt(r.diferenca)} | {fmt(r.herdado)} | "
                 f"{fmt(r.narrador)} |")
    if examples:
        L += ["", "## Exemplos", ""]
        for r in examples:
            cs = r["case"]
            L += [f"### {cs.ds.name}, {cs.modelo}, instância {cs.instancia} (classe: {cs.label})", "",
                  f"> {r['texto']}", ""]
            for c in r["llm"]:
                L.append(f"- {describe_claim(c)}")
                if c["tipo"] == "SUF" and c["falsa"]:
                    cex = llm.nearest_counterexample(cs, c["indices"])
                    if cex:
                        L.append(f"  - contraexemplo ({cex['dominio']}): mantidos "
                                 f"{', '.join(cex['mantidos'])}; alterados "
                                 f"{'; '.join(cex['alterados'])}; classe {cex['classe']}.")
                if c["tipo"] == "IRR" and c["falsa"]:
                    w = llm.axp_witness(cs, c["indices"][0])
                    if w:
                        L.append(f"  - o atributo está na explicação abdutiva {{{', '.join(w)}}}.")
            L.append("")
    L += ["## Leituras-padrão (autorização pela saída do SHAP)", ""]
    for t, text in llm.LEITURAS.items():
        L.append(f"- {t}: {text}" + (f" (tau = {args.tau:g})" if t == "IRR" else "") + ".")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def environment(args, narrator, n_cases, records, elapsed, interrupted, cache):
    import scipy
    import shap
    import sklearn
    info = narrator.info()
    return {
        "data": dt.date.today().isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "shap": shap.__version__,
        "narrador": info["narrador"], "extrator": info["extrator"],
        "estilo": None if args.backend == "mock" else args.style,
        "temperatura": 0.0, "semente": SEED, "num_ctx": args.num_ctx,
        "tokens_max": {"narracao": llm.NARRATION_TOKENS, "extracao": llm.EXTRACTION_TOKENS},
        "tau_irrelevancia": args.tau, "margem_nao_inferioridade": args.margin,
        "bases": args.bases or "todas", "modelos": args.models or list(MODELS),
        "instancias_por_base": args.instances, "casos_previstos": n_cases,
        "narrativas": len(records), "interrompido": interrupted,
        "chamadas_novas": narrator.new_calls,
        "respostas_do_cache": cache.hits if cache else 0,
        "tempo_execucao_s": round(elapsed, 1),
    }


def save_outputs(out, args, narrator, cases, records, elapsed, interrupted, cache):
    out.mkdir(parents=True, exist_ok=True)
    claims, narr = claim_table(records), narrative_table(records)
    bm, bt, bb = by_model(narr, claims), by_type(claims, narr), by_base(narr)
    ni = noninferiority(narr, args.margin)
    env = environment(args, narrator, len(cases), records, elapsed, interrupted, cache)
    nar_info = env["narrador"]
    run_info = dict(
        narrador=(f"{nar_info.get('modelo')}" + (f" ({nar_info.get('servidor')}"
                  + (f", {nar_info['parametros']}" if nar_info.get("parametros") else "")
                  + (f", {nar_info['quantizacao']}" if nar_info.get("quantizacao") else "")
                  + ")" if nar_info.get("servidor") else "")),
        extrator=env["extrator"] if isinstance(env["extrator"], str) else env["extrator"].get("modelo"),
        estilo=env["estilo"] or "não se aplica (sem LLM)", narrativas=len(records),
        casos_previstos=len(cases), interrompido=interrupted,
        falhas_extracao=int(narr.falha_extracao.sum()) if len(narr) else 0)

    claims.to_csv(out / "afirmacoes.csv", index=False)
    narr.to_csv(out / "por_narrativa.csv", index=False)
    bm.to_csv(out / "por_modelo_llm.csv", index=False)
    bt.to_csv(out / "por_tipo.csv", index=False)
    bb.to_csv(out / "por_base_llm.csv", index=False)
    ni.to_csv(out / "nao_inferioridade.csv", index=False)
    with (out / "narrativas.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(full_record(r), ensure_ascii=False, default=_jsonable) + "\n")
    if args.annotation_sample > 0 and records:
        annotation_sheet(records, args.annotation_sample).to_csv(
            out / "anotacao.csv", index=False, sep=";", encoding="utf-8-sig")
    write_latex(out / "tabela_llm.tex", bm, run_info)
    write_markdown(out / "resumo_llm.md", bm, bt, bb, ni, pick_examples(records), run_info, args)
    if cases:
        first = cases[0]
        prompts = dict(leituras_padrao=llm.LEITURAS, tau=args.tau,
                       narracao={s: llm.narration_messages(first, s) for s in llm.INSTRUCOES},
                       extracao=llm.extraction_messages(first, llm.verbalize_template(first)),
                       esquema=llm.extraction_schema(llm.display_names(first.ds)),
                       exemplo=first.key)
        (out / "instrucoes.json").write_text(json.dumps(prompts, indent=2, ensure_ascii=False)
                                             + "\n", encoding="utf-8")
    (out / "execucao.json").write_text(json.dumps(env, indent=2, ensure_ascii=False,
                                                  default=_jsonable) + "\n", encoding="utf-8")
    return bm, ni

def combine(paths, out):
    files = []
    for p in paths:
        for q in sorted(glob.glob(p)) or [p]:
            q = Path(q)
            if q.is_file() and q.name == "por_modelo_llm.csv":
                files.append(q)
            elif q.is_dir():
                files += sorted(q.rglob("por_modelo_llm.csv"))
    rows = []
    for f in dict.fromkeys(files):
        env_path = f.parent / "execucao.json"
        if not env_path.exists():
            continue
        env = json.loads(env_path.read_text(encoding="utf-8"))
        bm = pd.read_csv(f)
        ni_path = f.parent / "nao_inferioridade.csv"
        ni = pd.read_csv(ni_path) if ni_path.exists() else pd.DataFrame()
        t = bm[bm.modelo == "todos"]
        if not len(t):
            continue
        t = t.iloc[0]
        n = ni[ni.modelo == "todos"].iloc[0] if len(ni) and (ni.modelo == "todos").any() else None
        rows.append(dict(narrador=env["narrador"].get("modelo"), estilo=env.get("estilo") or "--",
                         narrativas=int(t.narrativas), afirmacoes_por_narrativa=t.afirmacoes_por_narrativa,
                         SUF_falsa=t.SUF_falsa, REL_falsa=t.REL_falsa, IRR_falsa=t.IRR_falsa,
                         CON_falsa=t.CON_falsa, alguma_falsa=t.alguma_falsa,
                         padrao_alguma_falsa=t.padrao_alguma_falsa, herdado=t.herdado,
                         narrador_erro=t.narrador,
                         diferenca_media=None if n is None else n.diferenca_media,
                         limite_inferior_95=None if n is None else n.limite_inferior_95,
                         nao_inferior=None if n is None else bool(n.nao_inferior),
                         pasta="/".join(f.parent.parts[-2:])))
    if not rows:
        print("Nenhum resultado encontrado (procure pastas com por_modelo_llm.csv).")
        return 1
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparacao_llms.csv", index=False)
    lines = [r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{4}{c}{Afirmações falsas (\%)} & \multicolumn{2}{c}{Alguma falsa (\%)}"
             r" & \multicolumn{2}{c}{Origem (\%)}\\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}",
             r"Narrador & Estilo & Narrativas & Suf. & Rel. & Irr. & Con. & LLM & Padrão & Herdado & Narrador\\",
             r"\midrule"]
    for _, r in df.iterrows():
        lines.append(f"{r.narrador.replace('_', '-')} & {r.estilo} & {r.narrativas} & {fmt(r.SUF_falsa)} & "
                     f"{fmt(r.REL_falsa)} & {fmt(r.IRR_falsa)} & {fmt(r.CON_falsa)} & "
                     f"{fmt(r.alguma_falsa)} & {fmt(r.padrao_alguma_falsa)} & {fmt(r.herdado)} & "
                     f"{fmt(r.narrador_erro)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tabela_comparacao_llms.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    show = df.drop(columns=["pasta"])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\nTabelas em '{out.as_posix()}/' (comparacao_llms.csv, tabela_comparacao_llms.tex).")
    return 0

def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    args = parse_args(argv)
    if args.combine:
        return combine(args.combine, Path(args.out))
    out = run_dir(args)
    cache = None if args.backend == "mock" else llm.Cache(out / "cache.jsonl")
    narrator = make_narrator(args, cache)
    try:
        info = narrator.info()
    except llm.LLMError as e:
        print(f"Erro: {e}")
        return 2
    print(f"Narrador: {info['narrador'].get('modelo')} ({info['narrador'].get('backend')}); "
          f"saída em '{out.as_posix()}/'.")
    print("Preparando as instâncias do piloto (modelos, SHAP e fatos formais)...", flush=True)
    cases = interleave(build_cases(bases=args.bases, models=args.models,
                                   instances=args.instances))
    if args.max_cases:
        cases = cases[:args.max_cases]
    print(f"{len(cases)} narrativas a gerar.", flush=True)
    records, elapsed, interrupted, error = run(args, narrator, cases)
    if not records:
        print("Nenhuma narrativa gerada.")
        return 2 if error else 1
    bm, ni = save_outputs(out, args, narrator, cases, records, elapsed, interrupted, cache)
    if interrupted:
        print(f"\nInterrompido após {fmt_duration(elapsed)}: {len(records)} de {len(cases)} "
              f"narrativas. Resultados parciais em '{out.as_posix()}/'.\n")
    else:
        print(f"\nConcluído em {fmt_duration(elapsed)}. Resultados em '{out.as_posix()}/'.\n")
    cols = ["modelo", "narrativas", "SUF_falsa", "REL_falsa", "IRR_falsa", "CON_falsa",
            "alguma_falsa", "padrao_alguma_falsa", "herdado", "narrador"]
    print(bm[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if len(ni):
        print()
        print(ni[["modelo", "bases", "diferenca_media", "limite_inferior_95", "margem",
                  "nao_inferior"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if error:
        return 2
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
