# Narrativas redigidas por LLM: resultados

Narrador: narrativa-padrao; extrator: o mesmo; estilo de instrução: não se aplica (sem LLM).
Narrativas: 54 de 54 previstas; falhas de extração: 0.
Vereditos no espaço de entrada completo, salvo indicação. A narrativa-padrão é a do piloto, avaliada nas mesmas instâncias.

## Afirmações falsas: LLM e narrativa-padrão

| Modelo | Narrativas | Afirmações/narr. | Verificáveis/narr. | Suf. | Rel. | Irr. | Con. | Alguma (LLM) | Alguma (padrão) | Alguma nos dados (LLM / padrão) | Herdado | Narrador |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| AD | 18 | 4.0 | 4.0 | 39 | 0 | 17 | 11 | 50 | 50 | 94 / 94 | 50 | 0 |
| RF | 18 | 4.0 | 4.0 | 33 | 0 | 39 | 22 | 50 | 50 | 89 / 89 | 50 | 0 |
| GB | 18 | 4.0 | 4.0 | 44 | 0 | 22 | 11 | 61 | 61 | 94 / 94 | 61 | 0 |
| todos | 54 | 4.0 | 4.0 | 39 | 0 | 26 | 15 | 54 | 54 | 93 / 93 | 54 | 0 |

Suf., Rel., Irr. e Con.: % das afirmações do LLM de cada tipo que são falsas. Alguma: % das narrativas com ao menos uma afirmação verificável falsa. Herdado e Narrador: % das narrativas com afirmação falsa autorizada e não autorizada pela saída do SHAP, respectivamente (uma narrativa pode ter as duas).

Mesmos tipos de afirmação na narrativa-padrão (% falsas): AD: Suf. 39, Rel. 0, Irr. 17, Con. 11; RF: Suf. 33, Rel. 0, Irr. 39, Con. 22; GB: Suf. 44, Rel. 0, Irr. 22, Con. 11; todos: Suf. 39, Rel. 0, Irr. 26, Con. 15.

Das afirmações verificáveis falsas do LLM, 100% eram autorizadas pela saída do SHAP (erro herdado da fonte) e 0% não eram (erro do narrador).

## Não inferioridade pareada por base (H1)

Margem fixada antes da execução: 10,0 pontos percentuais. Não inferior: o limite inferior unilateral de 95% da diferença média (LLM menos padrão), por reamostragem das bases, fica acima de menos a margem.

| Modelo | Bases | LLM (%) | Padrão (%) | Diferença média (p.p.) | Limite inferior 95% | IC 95% | Não inferior? | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---|---|---:|
| AD | 9 | 50 | 50 | 0.0 | 0.0 | 0.0 a 0.0 | sim | 0.002 |
| RF | 9 | 50 | 50 | 0.0 | 0.0 | 0.0 a 0.0 | sim | 0.002 |
| GB | 9 | 61 | 61 | 0.0 | 0.0 | 0.0 a 0.0 | sim | 0.002 |
| todos | 9 | 54 | 54 | 0.0 | 0.0 | 0.0 a 0.0 | sim | 0.002 |

## Cobertura da linguagem e fidelidade à saída do SHAP

| Tipo | Afirmações | % do total | Atributos reconhecidos (%) | Autorizadas pelo SHAP (%) | Falsas (%) | Falsas nos dados (%) | Atenuadas (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUF | 54 | 25 | 100 | 100 | 39 | 26 | 0 |
| REL | 54 | 25 | 100 | 100 | 0 | 0 | 0 |
| IRR | 54 | 25 | 100 | 100 | 26 | 81 | 0 |
| CON | 54 | 25 | 100 | 100 | 15 | 70 | 0 |
| ORD | 0 | 0 | -- | -- | -- | -- | -- |
| DIR | 0 | 0 | -- | -- | -- | -- | -- |
| OUTRA | 0 | 0 | -- | -- | -- | -- | -- |

ORD e DIR são conferidas só contra a saída do SHAP (fidelidade, como na literatura); OUTRA fica fora da linguagem. A cobertura indica quais extensões da linguagem (necessidade, contrafactuais com valores, ordem) mais fariam falta.

## Afirmações atenuadas e leitura branda da irrelevância

| Tipo | Atenuadas (n) | Falsas entre atenuadas (%) | Não atenuadas (n) | Falsas entre não atenuadas (%) |
|---|---:|---:|---:|---:|
| SUF | 0 | -- | 54 | 39 |
| REL | 0 | -- | 54 | 0 |
| IRR | 0 | -- | 54 | 26 |
| CON | 0 | -- | 54 | 15 |

Irrelevância lida de forma estrita (o atributo não pertence a nenhuma explicação abdutiva): 26% falsas. Lida de forma branda (mudar só esse atributo não reverteria a decisão): 2% falsas.

Suficiências falsas do LLM são reparadas acrescentando, em média, 1,33 atributos.

## Por base

| Base | Modelo | Narrativas | Alguma (LLM) | Alguma (padrão) | Diferença | Herdado | Narrador |
|---|---|---:|---:|---:|---:|---:|---:|
| corral | AD | 2 | 50 | 50 | 0 | 50 | 0 |
| corral | RF | 2 | 50 | 50 | 0 | 50 | 0 |
| corral | GB | 2 | 50 | 50 | 0 | 50 | 0 |
| mux6 | AD | 2 | 50 | 50 | 0 | 50 | 0 |
| mux6 | RF | 2 | 50 | 50 | 0 | 50 | 0 |
| mux6 | GB | 2 | 50 | 50 | 0 | 50 | 0 |
| parity5+5 | AD | 2 | 100 | 100 | 0 | 100 | 0 |
| parity5+5 | RF | 2 | 100 | 100 | 0 | 100 | 0 |
| parity5+5 | GB | 2 | 100 | 100 | 0 | 100 | 0 |
| monk1 | AD | 2 | 50 | 50 | 0 | 50 | 0 |
| monk1 | RF | 2 | 0 | 0 | 0 | 0 | 0 |
| monk1 | GB | 2 | 50 | 50 | 0 | 50 | 0 |
| monk2 | AD | 2 | 100 | 100 | 0 | 100 | 0 |
| monk2 | RF | 2 | 100 | 100 | 0 | 100 | 0 |
| monk2 | GB | 2 | 100 | 100 | 0 | 100 | 0 |
| monk3 | AD | 2 | 0 | 0 | 0 | 0 | 0 |
| monk3 | RF | 2 | 0 | 0 | 0 | 0 | 0 |
| monk3 | GB | 2 | 0 | 0 | 0 | 0 | 0 |
| breast-cancer | AD | 2 | 0 | 0 | 0 | 0 | 0 |
| breast-cancer | RF | 2 | 0 | 0 | 0 | 0 | 0 |
| breast-cancer | GB | 2 | 100 | 100 | 0 | 100 | 0 |
| wine | AD | 2 | 100 | 100 | 0 | 100 | 0 |
| wine | RF | 2 | 100 | 100 | 0 | 100 | 0 |
| wine | GB | 2 | 100 | 100 | 0 | 100 | 0 |
| iris | AD | 2 | 0 | 0 | 0 | 0 | 0 |
| iris | RF | 2 | 50 | 50 | 0 | 50 | 0 |
| iris | GB | 2 | 0 | 0 | 0 | 0 | 0 |

## Exemplos

### breast-cancer, GB, instância 0 (classe: maligno)

> O modelo classificou o caso como maligno. A decisão se explica por raio (pior) na faixa alta, área (pior) na faixa alta e perímetro (pior) na faixa alta. Raio (pior) influenciou a decisão e, se seu valor fosse diferente, a decisão poderia mudar. Já pontos côncavos (média) não influenciou o resultado.

- SUF(raio (pior), perímetro (pior), área (pior)) "a decisão se explica por": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- REL(raio (pior)) "influenciou a decisão": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- IRR(pontos côncavos (média)) "não influenciou o resultado": **falsa**; nos dados, falsa; autorizada pelo SHAP
  - o atributo está na explicação abdutiva {pontos côncavos (média), raio (pior), área (pior), pontos côncavos (pior)}.
- CON(raio (pior)) "se seu valor fosse diferente, a decisão poderia mudar": verdadeira; nos dados, falsa; autorizada pelo SHAP

### wine, AD, instância 0 (classe: cultivar 0)

> O modelo classificou o caso como cultivar 0. A decisão se explica por prolina na faixa alta, flavonoides na faixa alta e teor alcoólico na faixa alta. Prolina influenciou a decisão e, se seu valor fosse diferente, a decisão poderia mudar. Já alcalinidade das cinzas não influenciou o resultado.

- SUF(teor alcoólico, flavonoides, prolina) "a decisão se explica por": **falsa**; nos dados, verdadeira; autorizada pelo SHAP
  - contraexemplo (espaço completo): mantidos teor alcoólico na faixa alta, flavonoides na faixa alta, prolina na faixa alta; alterados intensidade de cor na faixa baixa (em vez de faixa alta); razão OD280/OD315 na faixa alta (em vez de faixa intermediária); classe outros cultivares.
- REL(prolina) "influenciou a decisão": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- IRR(alcalinidade das cinzas) "não influenciou o resultado": verdadeira; nos dados, falsa; autorizada pelo SHAP
- CON(prolina) "se seu valor fosse diferente, a decisão poderia mudar": verdadeira; nos dados, falsa; autorizada pelo SHAP

### wine, GB, instância 0 (classe: cultivar 0)

> O modelo classificou o caso como cultivar 0. A decisão se explica por prolina na faixa alta, teor alcoólico na faixa alta e flavonoides na faixa alta. Prolina influenciou a decisão e, se seu valor fosse diferente, a decisão poderia mudar. Já matiz não influenciou o resultado.

- SUF(teor alcoólico, flavonoides, prolina) "a decisão se explica por": **falsa**; nos dados, verdadeira; autorizada pelo SHAP
  - contraexemplo (espaço completo): mantidos teor alcoólico na faixa alta, flavonoides na faixa alta, prolina na faixa alta; alterados fenóis totais na faixa baixa (em vez de faixa alta); intensidade de cor na faixa baixa (em vez de faixa alta); classe outros cultivares.
- REL(prolina) "influenciou a decisão": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- IRR(matiz) "não influenciou o resultado": verdadeira; nos dados, falsa; autorizada pelo SHAP
- CON(prolina) "se seu valor fosse diferente, a decisão poderia mudar": verdadeira; nos dados, falsa; autorizada pelo SHAP

## Leituras-padrão (autorização pela saída do SHAP)

- SUF: todos os atributos citados têm contribuição SHAP positiva (a favor da decisão).
- REL: o atributo tem contribuição SHAP não nula.
- IRR: a contribuição do atributo é desprezível (|phi_i| <= tau * max_j |phi_j|) ou é a de menor módulo entre os atributos fora dos três que mais favorecem a decisão (tau = 0.1).
- CON: algum atributo citado tem contribuição SHAP positiva.
- ORD: os atributos citados estão em ordem não crescente de |phi|; um atributo citado sozinho tem o maior |phi| ou o maior phi.
- DIR: o sinal de phi_i coincide com o sentido afirmado (a favor: phi_i > 0; contra: phi_i < 0).
