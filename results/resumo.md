# Resultados do estudo piloto

Percentual de narrativas-padrão derivadas do SHAP com afirmação formalmente falsa (verificação exata no espaço de entrada completo).

## Agregado por modelo

| Modelo | Instâncias | Suficiência (top-3) | Relevância (top-1) | Irrelevância (último) | Contraste (top-1) | Alguma falsa | IC 95% (alguma) | Faixa por base |
|---|---:|---:|---:|---:|---:|---:|---|---|
| AD | 389 | 35 | 0 | 33 | 18 | 56 | 51–60 | 6–100 |
| RF | 389 | 26 | 0 | 58 | 33 | 65 | 60–70 | 14–100 |
| GB | 389 | 33 | 0 | 39 | 22 | 60 | 55–65 | 4–98 |

## Mesmas afirmações decididas sobre os dados de treino

| Modelo | Suficiência | Relevância | Irrelevância | Contraste | Alguma falsa |
|---|---:|---:|---:|---:|---:|
| AD | 26 | 1 | 72 | 54 | 80 |
| RF | 20 | 0 | 72 | 57 | 81 |
| GB | 23 | 0 | 69 | 54 | 78 |

## Custo da narrativa verificada

| Modelo | AXp mín. (média) | AXp mín. ≤ 3 (%) | CXp mín. (média) | CXp unitária (%) | Reparo médio da suficiência | Reparo = 1 (%) | Com atributo irrelevante (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AD | 2.78 | 77 | 1.12 | 88 | 1.32 | 70 | 89 |
| RF | 2.79 | 82 | 1.36 | 70 | 1.80 | 40 | 67 |
| GB | 2.75 | 76 | 1.22 | 81 | 1.53 | 60 | 84 |

## Indicadores adicionais

| Modelo | Atributos citados (média) | Irrelevância falsa, dado que existe irrelevante (%) | Suficiência falsa com contraexemplo nos dados (%) | Veredito muda com o domínio: Suf. / Rel. / Irr. / Con. (%) |
|---|---:|---:|---:|---|
| AD | 2.93 | 25 | 72 | 10 / 1 / 42 / 36 |
| RF | 2.85 | 37 | 77 | 6 / 0 / 25 / 24 |
| GB | 2.94 | 27 | 69 | 10 / 0 / 35 / 32 |

## Sensibilidade ao número de atributos citados (suficiência falsa, %)

| Modelo | suf_top1_falsa | suf_top2_falsa | suf_top3_falsa | suf_topkstar_falsa | suf_topm1_abs_falsa |
|---|---:|---:|---:|---:|---:|
| AD | 91 | 67 | 35 | 31 | 44 |
| RF | 95 | 57 | 26 | 25 | 35 |
| GB | 85 | 58 | 33 | 29 | 42 |

## Por base

| Base | Modelo | m | Acurácia | Suf. | Rel. | Irr. | Con. | Alguma |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| corral | AD | 6 | 0.98 | 0 | 0 | 3 | 13 | 16 |
| corral | RF | 6 | 0.98 | 0 | 0 | 19 | 13 | 32 |
| corral | GB | 6 | 1.00 | 0 | 0 | 0 | 13 | 13 |
| mux6 | AD | 6 | 0.92 | 23 | 0 | 67 | 3 | 67 |
| mux6 | RF | 6 | 1.00 | 27 | 0 | 60 | 0 | 63 |
| mux6 | GB | 6 | 1.00 | 33 | 0 | 50 | 0 | 63 |
| parity5+5 | AD | 10 | 0.52 | 100 | 0 | 56 | 2 | 100 |
| parity5+5 | RF | 10 | 0.49 | 100 | 0 | 92 | 20 | 100 |
| parity5+5 | GB | 10 | 0.47 | 94 | 0 | 78 | 12 | 98 |
| monk1 | AD | 6 | 0.79 | 8 | 0 | 8 | 10 | 26 |
| monk1 | RF | 6 | 0.94 | 4 | 0 | 10 | 4 | 14 |
| monk1 | GB | 6 | 0.77 | 36 | 0 | 26 | 4 | 42 |
| monk2 | AD | 6 | 0.65 | 70 | 0 | 68 | 26 | 88 |
| monk2 | RF | 6 | 0.66 | 30 | 0 | 90 | 74 | 96 |
| monk2 | GB | 6 | 0.65 | 26 | 0 | 70 | 46 | 78 |
| monk3 | AD | 6 | 0.95 | 2 | 0 | 4 | 2 | 6 |
| monk3 | RF | 6 | 0.93 | 4 | 0 | 18 | 2 | 24 |
| monk3 | GB | 6 | 0.97 | 0 | 0 | 0 | 4 | 4 |
| breast-cancer | AD | 8 | 0.95 | 40 | 0 | 10 | 20 | 44 |
| breast-cancer | RF | 8 | 0.96 | 14 | 0 | 86 | 52 | 88 |
| breast-cancer | GB | 8 | 0.96 | 50 | 0 | 60 | 24 | 84 |
| wine | AD | 8 | 0.94 | 43 | 0 | 49 | 59 | 92 |
| wine | RF | 8 | 0.98 | 29 | 0 | 80 | 78 | 100 |
| wine | GB | 8 | 0.94 | 29 | 0 | 20 | 63 | 92 |
| iris | AD | 4 | 0.98 | 0 | 0 | 38 | 24 | 48 |
| iris | RF | 4 | 0.98 | 10 | 0 | 45 | 34 | 52 |
| iris | GB | 4 | 0.96 | 7 | 0 | 31 | 21 | 41 |

## Exemplo

Base: breast-cancer; modelo: RF.

**Narrativa-padrão (SHAP):** O modelo classificou o caso como benigno. A decisão se explica por raio (pior) na faixa baixa, área (pior) na faixa baixa e perímetro (pior) na faixa intermediária. Raio (pior) influenciou a decisão e, se seu valor fosse diferente, a decisão poderia mudar. Já raio (média) não influenciou o resultado.

**Contraexemplo (dados de treino):** mantidos raio (pior) na faixa baixa, área (pior) na faixa baixa, perímetro (pior) na faixa intermediária; alterados concavidade (média) na faixa alta (em vez de intermediária); pontos côncavos (média) na faixa alta (em vez de intermediária); pontos côncavos (pior) na faixa alta (em vez de intermediária); classe maligno.

**Narrativa verificada:** O modelo classificou o caso como benigno. Estes valores bastam para essa decisão: raio (pior) na faixa baixa, área (pior) na faixa baixa e pontos côncavos (pior) na faixa intermediária; qualquer caso com eles recebe a mesma classe. Nenhuma mudança em um único atributo reverte a decisão; seria preciso alterar ao menos dois atributos ao mesmo tempo, por exemplo raio (pior) e área (pior).
