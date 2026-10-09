import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from verif import claims as C
from verif import datasets
from verif.experiment import MODELS, REAL, SEED, split_and_select
from verif.oracle import full_space, sample_space
from verif.shap_tool import shap_for_predicted

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CLAIMS = ["N1_suf_topk", "N2_rel_top1", "N3_irr_ultimo", "N4_con_top1"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instances", type=int, default=50,
                   help="instâncias de teste explicadas por base (padrão: 50)")
    p.add_argument("--k", type=int, default=3,
                   help="atributos citados na afirmação de suficiência (padrão: 3)")
    p.add_argument("--out", default="results", help="pasta de saída (padrão: results)")
    p.add_argument("--quick", action="store_true", help="execução reduzida")
    args = p.parse_args()
    if args.quick:
        args.instances = 10
    return args

def run(args):
    rows, model_rows, cases = [], [], []
    for ds in datasets.load_all(SEED):
        X_tr, X_te, y_tr, y_te, explain, background = split_and_select(ds, args.instances)

        for model_name, make_model in MODELS.items():
            model = make_model().fit(X_tr, y_tr)
            full = full_space(model.predict, ds.domains)
            data = sample_space(model.predict, X_tr)
            predicted = model.predict(explain).astype(int)
            phi = shap_for_predicted(model, background, explain, predicted)
            model_rows.append(dict(base=ds.name, origem=ds.origin, modelo=model_name, m=ds.m,
                                   espaco=full.n_points, pontos_treino=data.n_points,
                                   acuracia=model.score(X_te, y_te), instancias=len(explain)))

            for k, v in enumerate(explain):
                c = int(predicted[k])
                f_full = C.Facts(full.waxp(v, c), ds.m)
                if not f_full.decision_depends_on_inputs: 
                    continue
                f_data = C.Facts(data.waxp(v, c), ds.m)
                narrative = C.shap_narrative(phi[k], k=min(args.k, ds.m - 1))
                verified = C.verified_narrative(f_full, phi[k])
                order = C.shap_order(phi[k])
                axp_min = int(f_full.sizes[f_full.smallest_axp()])
                cxp_min = int(f_full.sizes[f_full.smallest_cxp()])
                top_mask = narrative["N1_suf_topk"].mask
                k_used = len(narrative["N1_suf_topk"].features)

                row = dict(base=ds.name, modelo=model_name, instancia=k, m=ds.m, classe=c,
                           n_relevantes=int(f_full.relevant().sum()),
                           n_axps=len(f_full.axps()), n_cxps=len(f_full.cxps()),
                           axp_min=axp_min, cxp_min=cxp_min,
                           existe_irrelevante=bool((~f_full.relevant()).any()),
                           reparo_topk=f_full.repair_distance(top_mask), k_usado=k_used,
                           positivos=int((phi[k] > 0).sum()),
                           dados_constante=not f_data.decision_depends_on_inputs)
                for name in CLAIMS:
                    row[f"{name}_falsa"] = not narrative[name].holds(f_full)
                    row[f"{name}_falsa_dados"] = not narrative[name].holds(f_data)
                row["alguma_falsa"] = any(row[f"{n}_falsa"] for n in CLAIMS)
                row["alguma_falsa_dados"] = any(row[f"{n}_falsa_dados"] for n in CLAIMS)
                for kk in (1, 2, 3):
                    if kk < ds.m:
                        row[f"suf_top{kk}_falsa"] = not f_full.sufficient(
                            C.to_mask(C.favorable(phi[k], kk)))
                row["suf_topkstar_falsa"] = not f_full.sufficient(
                    C.to_mask(C.favorable(phi[k], axp_min)))
                row["suf_topm1_abs_falsa"] = not f_full.sufficient(C.to_mask(order[:3]))
                rows.append(row)

                if ds.name in REAL:
                    cases.append(dict(ds=ds, model=model_name, v=v.copy(), c=c, phi=phi[k].copy(),
                                      narrative=narrative, verified=verified,
                                      f_full=f_full, f_data=f_data, data=data, full=full,
                                      row=row))
        print(f"  {ds.name:<14s} m={ds.m:<2d} |F|={ds.space_size:<6d} ok", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(model_rows), cases

FALSE_COLS = [f"{n}_falsa" for n in CLAIMS] + ["alguma_falsa"]
DATA_COLS = [f"{n}_falsa_dados" for n in CLAIMS] + ["alguma_falsa_dados"]


def wilson(p, n, z=1.96):
    """Intervalo de Wilson (95%) para uma proporção."""
    if n == 0:
        return (np.nan, np.nan)
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)


def summarize(df, mdl):
    by_base = (df.groupby(["base", "modelo"], sort=False)[FALSE_COLS]
                 .mean().mul(100).reset_index())
    by_base["instancias"] = df.groupby(["base", "modelo"], sort=False).size().values
    by_base = by_base.merge(mdl[["base", "modelo", "m", "espaco", "acuracia"]],
                            on=["base", "modelo"])

    pooled = df.groupby("modelo", sort=False)[FALSE_COLS + DATA_COLS].mean().mul(100)
    pooled["instancias"] = df.groupby("modelo", sort=False).size()
    for col in FALSE_COLS:
        lo, hi = zip(*[wilson(p / 100, n) for p, n in zip(pooled[col], pooled["instancias"])])
        pooled[f"{col}_ic_inf"] = np.array(lo) * 100
        pooled[f"{col}_ic_sup"] = np.array(hi) * 100
    span = by_base.groupby("modelo", sort=False)["alguma_falsa"].agg(["min", "max"])
    pooled["alguma_falsa_min_base"] = span["min"]
    pooled["alguma_falsa_max_base"] = span["max"]

    cost = df.groupby("modelo", sort=False).agg(
        axp_min_media=("axp_min", "mean"),
        axp_min_ate3=("axp_min", lambda s: (s <= 3).mean() * 100),
        cxp_min_media=("cxp_min", "mean"),
        cxp_unitaria=("cxp_min", lambda s: (s == 1).mean() * 100),
        com_irrelevante=("existe_irrelevante", lambda s: s.mean() * 100),
        dados_constante=("dados_constante", lambda s: s.mean() * 100),
    )
    false_suf = df[df["N1_suf_topk_falsa"]]
    cost["reparo_medio"] = false_suf.groupby("modelo", sort=False)["reparo_topk"].mean()
    cost["reparo_1"] = false_suf.groupby("modelo", sort=False)["reparo_topk"].apply(
        lambda s: (s == 1).mean() * 100)

    cond = df[df["existe_irrelevante"]]
    cost["irr_falsa_condicional"] = cond.groupby("modelo", sort=False)["N3_irr_ultimo_falsa"].mean() * 100
    cost["k_medio"] = df.groupby("modelo", sort=False)["k_usado"].mean()
    suf_false = df[df["N1_suf_topk_falsa"]]
    cost["contraexemplo_nos_dados"] = (suf_false.groupby("modelo", sort=False)
                                       ["N1_suf_topk_falsa_dados"].mean() * 100)
    for name in CLAIMS:
        cost[f"muda_com_dominio_{name}"] = (df[f"{name}_falsa"] != df[f"{name}_falsa_dados"]).groupby(
            df["modelo"], sort=False).mean() * 100

    sens_cols = [c for c in ("suf_top1_falsa", "suf_top2_falsa", "suf_top3_falsa",
                             "suf_topkstar_falsa", "suf_topm1_abs_falsa") if c in df]
    sens = df.groupby("modelo", sort=False)[sens_cols].mean().mul(100)
    df = df.assign(grupo=np.where(df.base.isin(REAL), "reais", "sinteticas"))
    agg = {c: "mean" for c in FALSE_COLS + DATA_COLS}
    by_group = df.groupby(["grupo", "modelo"], sort=False).agg(agg).mul(100).reset_index()
    sf = df[df["N1_suf_topk_falsa"]]
    cx = (sf.groupby(["grupo", "modelo"], sort=False)["N1_suf_topk_falsa_dados"].mean().mul(100)
            .rename("contraexemplo_nos_dados").reset_index())
    by_group = by_group.merge(cx, on=["grupo", "modelo"], how="left")
    return by_base, pooled, cost, sens, by_group

def hamming(a, b):
    return int(np.sum(np.asarray(a) != np.asarray(b)))


def pick_example(cases):
    """Instância real, explicada por um ensemble, em que a afirmação de suficiência é
    refutada por um ponto dos próprios dados de treino (contraexemplo realista), com o
    maior número de afirmações falsas, explicação verificada curta e o contraexemplo
    mais próximo da instância."""
    best, best_key = None, None
    priority = {name: i for i, name in enumerate(REAL)}
    for case in cases:
        row = case["row"]
        if case["model"] not in ("RF", "GB"):
            continue
        if not (row["N1_suf_topk_falsa"] and row["N1_suf_topk_falsa_dados"]):
            continue
        mask = case["narrative"]["N1_suf_topk"].mask
        cex = case["data"].counterexamples(case["v"], case["c"], mask)
        if len(cex) == 0:
            continue
        dist = [hamming(x, case["v"]) for x in cex]
        j = int(np.argmin(dist))
        n_false = sum(row[f"{n}_falsa"] for n in CLAIMS)
        key = (-n_false, row["axp_min"] > 4, dist[j], row["axp_min"],
               priority[case["ds"].name], case["model"] != "RF")
        if best_key is None or key < best_key:
            best, best_key = dict(case, cex=cex[j]), key
    return best


def example_payload(ex):
    ds, v, c = ex["ds"], ex["v"], ex["c"]
    label = ds.class_names[c]
    other = ds.class_names[1 - c]
    nar, ver = ex["narrative"], ex["verified"]
    verdicts = {name: dict(afirmacao=nar[name].kind, atributos=[ds.feature_names[i] for i in nar[name].features],
                           verdadeira_espaco=nar[name].holds(ex["f_full"]),
                           verdadeira_dados=nar[name].holds(ex["f_data"]))
                for name in CLAIMS}
    cex = ex["cex"]
    diff = [i for i in range(ds.m) if cex[i] != v[i]]
    return dict(
        base=ds.name, modelo=ex["model"], classe=label, classe_oposta=other,
        instancia={ds.feature_names[i]: ds.value_label(i, v[i]) for i in range(ds.m)},
        shap={ds.feature_names[i]: round(float(ex["phi"][i]), 4)
              for i in np.argsort(-np.asarray(ex["phi"]), kind="stable")},
        narrativa_shap=C.verbalize_shap(nar, ds, v, label),
        vereditos=verdicts,
        contraexemplo=dict(
            mantidos=[ds.describe_value(i, v[i]) for i in nar["N1_suf_topk"].features],
            alterados=[f"{ds.feature_names[i]} na faixa {ds.value_label(i, cex[i])} "
                       f"(em vez de {ds.value_label(i, v[i])})" for i in diff],
            classe=other),
        narrativa_verificada=C.verbalize_verified(ver, ds, v, label),
        afirmacoes_verificadas={name: [ds.feature_names[i] for i in cl.features]
                                for name, cl in ver.items()},
    )


def latex_escape(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
             .replace("_", r"\_").replace("#", r"\#").replace("→", r"$\rightarrow$"))


def write_example_tex(path, p):
    names = {"N1_suf_topk": "suficiência de", "N2_rel_top1": "relevância de",
             "N3_irr_ultimo": "irrelevância de", "N4_con_top1": "contraste em"}
    ver = []
    for name, d in p["vereditos"].items():
        ok = d["verdadeira_espaco"]
        mark = "verdadeira" if ok else r"\textbf{falsa}"
        ver.append(f"{names[name]} {latex_escape(C._join(d['atributos']))}: {mark}")
    ver[0] = ver[0][:1].upper() + ver[0][1:]
    cex = p["contraexemplo"]
    cex_text = (f"um caso dos dados de treino com {latex_escape(C._join(cex['mantidos']))}, "
                f"mas com {latex_escape(C._join(cex['alterados']))}, é classificado como "
                f"{latex_escape(cex['classe'])}")
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{3.1cm}>{\raggedright\arraybackslash}X@{}}",
        r"\toprule",
        rf"Narrativa-padrão (SHAP) & \textit{{{latex_escape(p['narrativa_shap'])}}}\\",
        r"\addlinespace",
        r"Veredito do verificador & " + "; ".join(ver) + r".\\",
        r"\addlinespace",
        r"Contraexemplo à suficiência & " + cex_text[0].upper() + cex_text[1:] + r".\\",
        r"\addlinespace",
        rf"Narrativa verificada & \textit{{{latex_escape(p['narrativa_verificada'])}}}\\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def pct(x):
    return f"{x:.0f}"


def br(x, nd=1):
    """Número com vírgula decimal (padrão brasileiro), para as tabelas em LaTeX."""
    return f"{x:.{nd}f}".replace(".", ",")


def write_latex(out, by_base, pooled, cost, by_group):
    rf = by_base[by_base.modelo == "RF"]
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{4}{c}{Afirmação falsa (\% das narrativas)} & \\",
             r"\cmidrule(lr){4-7}",
             r"Base & $m$ & Acurácia & Suficiência & Relevância & Irrelevância & Contraste & Alguma\\",
             r"\midrule"]
    for _, r in rf.iterrows():
        lines.append(f"{r.base} & {r.m} & {br(r.acuracia, 2)} & {pct(r.N1_suf_topk_falsa)} & "
                     f"{pct(r.N2_rel_top1_falsa)} & {pct(r.N3_irr_ultimo_falsa)} & "
                     f"{pct(r.N4_con_top1_falsa)} & {pct(r.alguma_falsa)}\\\\")
    lines.append(r"\midrule")
    labels = {"AD": "árvores de decisão", "RF": "\\textit{random forests}", "GB": "\\textit{boosting}"}
    for model in ("RF", "AD", "GB"):
        p = pooled.loc[model]
        lines.append(rf"\multicolumn{{3}}{{@{{}}l}}{{Todas as bases, {labels[model]}}} & "
                     f"{pct(p.N1_suf_topk_falsa)} & {pct(p.N2_rel_top1_falsa)} & "
                     f"{pct(p.N3_irr_ultimo_falsa)} & {pct(p.N4_con_top1_falsa)} & "
                     f"{pct(p.alguma_falsa)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tabela_narrativas.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    names = {"AD": "Árvore de decisão", "RF": "\\textit{Random forest}", "GB": "\\textit{Boosting}"}
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{AXp mínima} & \multicolumn{2}{c}{CXp mínima} & "
             r"\multicolumn{2}{c}{Reparo da suficiência}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"Modelo & Tamanho & $\le3$ (\%) & Tamanho & Unitária (\%) & Atributos & Basta um (\%)\\",
             r"\midrule"]
    for model in ("AD", "RF", "GB"):
        q = cost.loc[model]
        lines.append(f"{names[model]} & {br(q.axp_min_media)} & {pct(q.axp_min_ate3)} & "
                     f"{br(q.cxp_min_media)} & {pct(q.cxp_unitaria)} & {br(q.reparo_medio)} & "
                     f"{pct(q.reparo_1)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tabela_verificadas.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Suficiência} & \multicolumn{2}{c}{Irrelevância} & "
             r"\multicolumn{2}{c}{Contraste} & Refutada por\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Modelo & Espaço & Dados & Espaço & Dados & Espaço & Dados & caso dos dados (\%)\\"]
    for group, label in (("sinteticas", "Conceitos sintéticos"), ("reais", "Bases reais")):
        g = by_group[by_group.grupo == group].set_index("modelo")
        lines += [r"\midrule", rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{label}}}}}\\"]
        for model in ("AD", "RF", "GB"):
            q = g.loc[model]
            lines.append(f"{names[model]} & {pct(q.N1_suf_topk_falsa)} & {pct(q.N1_suf_topk_falsa_dados)} & "
                         f"{pct(q.N3_irr_ultimo_falsa)} & {pct(q.N3_irr_ultimo_falsa_dados)} & "
                         f"{pct(q.N4_con_top1_falsa)} & {pct(q.N4_con_top1_falsa_dados)} & "
                         f"{pct(q.contraexemplo_nos_dados)}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tabela_dominio.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path, by_base, pooled, cost, sens, example):
    L = ["# Resultados do estudo piloto", "",
         "Percentual de narrativas-padrão derivadas do SHAP com afirmação formalmente falsa "
         "(verificação exata no espaço de entrada completo).", "",
         "## Agregado por modelo", "",
         "| Modelo | Instâncias | Suficiência (top-3) | Relevância (top-1) | Irrelevância (último) "
         "| Contraste (top-1) | Alguma falsa | IC 95% (alguma) | Faixa por base |",
         "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for model, p in pooled.iterrows():
        L.append(f"| {model} | {int(p.instancias)} | {pct(p.N1_suf_topk_falsa)} | "
                 f"{pct(p.N2_rel_top1_falsa)} | {pct(p.N3_irr_ultimo_falsa)} | "
                 f"{pct(p.N4_con_top1_falsa)} | {pct(p.alguma_falsa)} | "
                 f"{pct(p.alguma_falsa_ic_inf)}–{pct(p.alguma_falsa_ic_sup)} | "
                 f"{pct(p.alguma_falsa_min_base)}–{pct(p.alguma_falsa_max_base)} |")
    L += ["", "## Mesmas afirmações decididas sobre os dados de treino", "",
          "| Modelo | Suficiência | Relevância | Irrelevância | Contraste | Alguma falsa |",
          "|---|---:|---:|---:|---:|---:|"]
    for model, p in pooled.iterrows():
        L.append(f"| {model} | {pct(p.N1_suf_topk_falsa_dados)} | {pct(p.N2_rel_top1_falsa_dados)} | "
                 f"{pct(p.N3_irr_ultimo_falsa_dados)} | {pct(p.N4_con_top1_falsa_dados)} | "
                 f"{pct(p.alguma_falsa_dados)} |")
    L += ["", "## Custo da narrativa verificada", "",
          "| Modelo | AXp mín. (média) | AXp mín. ≤ 3 (%) | CXp mín. (média) | CXp unitária (%) "
          "| Reparo médio da suficiência | Reparo = 1 (%) | Com atributo irrelevante (%) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, q in cost.iterrows():
        L.append(f"| {model} | {q.axp_min_media:.2f} | {pct(q.axp_min_ate3)} | {q.cxp_min_media:.2f} | "
                 f"{pct(q.cxp_unitaria)} | {q.reparo_medio:.2f} | {pct(q.reparo_1)} | "
                 f"{pct(q.com_irrelevante)} |")
    L += ["", "## Indicadores adicionais", "",
          "| Modelo | Atributos citados (média) | Irrelevância falsa, dado que existe irrelevante (%) "
          "| Suficiência falsa com contraexemplo nos dados (%) | Veredito muda com o domínio: "
          "Suf. / Rel. / Irr. / Con. (%) |",
          "|---|---:|---:|---:|---|"]
    for model, q in cost.iterrows():
        mud = " / ".join(pct(q[f"muda_com_dominio_{n}"]) for n in CLAIMS)
        L.append(f"| {model} | {q.k_medio:.2f} | {pct(q.irr_falsa_condicional)} | "
                 f"{pct(q.contraexemplo_nos_dados)} | {mud} |")
    L += ["", "## Sensibilidade ao número de atributos citados (suficiência falsa, %)", "",
          "| Modelo | " + " | ".join(sens.columns) + " |",
          "|---|" + "---:|" * len(sens.columns)]
    for model, s in sens.iterrows():
        L.append(f"| {model} | " + " | ".join(pct(x) for x in s.values) + " |")
    L += ["", "## Por base", "",
          "| Base | Modelo | m | Acurácia | Suf. | Rel. | Irr. | Con. | Alguma |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in by_base.iterrows():
        L.append(f"| {r.base} | {r.modelo} | {r.m} | {r.acuracia:.2f} | {pct(r.N1_suf_topk_falsa)} | "
                 f"{pct(r.N2_rel_top1_falsa)} | {pct(r.N3_irr_ultimo_falsa)} | "
                 f"{pct(r.N4_con_top1_falsa)} | {pct(r.alguma_falsa)} |")
    if example:
        L += ["", "## Exemplo", "", f"Base: {example['base']}; modelo: {example['modelo']}.", "",
              f"**Narrativa-padrão (SHAP):** {example['narrativa_shap']}", "",
              "**Contraexemplo (dados de treino):** mantidos "
              + ", ".join(example["contraexemplo"]["mantidos"]) + "; alterados "
              + "; ".join(example["contraexemplo"]["alterados"])
              + f"; classe {example['contraexemplo']['classe']}.", "",
              f"**Narrativa verificada:** {example['narrativa_verificada']}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def environment(args, elapsed):
    import scipy
    import shap
    import sklearn
    return {
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "shap": shap.__version__,
        "semente": SEED, "instancias_por_base": args.instances, "k": args.k,
        "tempo_execucao_s": round(elapsed, 1),
    }


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Executando o estudo piloto...")
    t0 = time.time()
    df, mdl, cases = run(args)
    by_base, pooled, cost, sens, by_group = summarize(df, mdl)
    example = pick_example(cases)
    payload = example_payload(example) if example else None
    elapsed = time.time() - t0

    df.to_csv(out / "por_instancia.csv", index=False)
    by_base.to_csv(out / "por_base.csv", index=False)
    pooled.to_csv(out / "por_modelo.csv")
    cost.to_csv(out / "custo_verificadas.csv")
    sens.to_csv(out / "sensibilidade_k.csv")
    mdl.to_csv(out / "modelos.csv", index=False)
    write_latex(out, by_base, pooled, cost, by_group)
    by_group.to_csv(out / "por_grupo.csv", index=False)
    if payload:
        (out / "exemplo.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        write_example_tex(out / "exemplo.tex", payload)
    write_markdown(out / "resumo.md", by_base, pooled, cost, sens, payload)
    (out / "ambiente.json").write_text(json.dumps(environment(args, elapsed), indent=2,
                                                  ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nConcluído em {elapsed:.1f} s. Resultados em '{out}/'.\n")
    show = pooled[[c for c in FALSE_COLS + DATA_COLS] + ["instancias"]]
    print(show.to_string(float_format=lambda v: f"{v:.1f}"))
    print()
    print(cost.to_string(float_format=lambda v: f"{v:.2f}"))
    print()
    print(sens.to_string(float_format=lambda v: f"{v:.1f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
