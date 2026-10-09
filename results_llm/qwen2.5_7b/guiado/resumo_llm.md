# Narrativas redigidas por LLM: resultados

Narrador: qwen2.5:7b (Ollama 0.30.11, 7.6B, Q4_K_M); extrator: o mesmo; estilo de instrução: guiado.
Narrativas: 6 de 6 previstas; falhas de extração: 0.
Vereditos no espaço de entrada completo, salvo indicação. A narrativa-padrão é a do piloto, avaliada nas mesmas instâncias.

## Afirmações falsas: LLM e narrativa-padrão

| Modelo | Narrativas | Afirmações/narr. | Verificáveis/narr. | Suf. | Rel. | Irr. | Con. | Alguma (LLM) | Alguma (padrão) | Alguma nos dados (LLM / padrão) | Herdado | Narrador |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| AD | 2 | 8.5 | 6.5 | 0 | 14 | 0 | 33 | 50 | 50 | 100 / 100 | 50 | 50 |
| RF | 2 | 7.5 | 6.5 | 50 | 14 | 50 | 0 | 100 | 50 | 100 / 100 | 50 | 50 |
| GB | 2 | 10.0 | 6.5 | 50 | 0 | 0 | 33 | 50 | 50 | 100 / 100 | 50 | 0 |
| todos | 6 | 8.7 | 6.5 | 40 | 10 | 17 | 25 | 67 | 50 | 100 / 100 | 50 | 33 |

Suf., Rel., Irr. e Con.: % das afirmações do LLM de cada tipo que são falsas. Alguma: % das narrativas com ao menos uma afirmação verificável falsa. Herdado e Narrador: % das narrativas com afirmação falsa autorizada e não autorizada pela saída do SHAP, respectivamente (uma narrativa pode ter as duas).

Mesmos tipos de afirmação na narrativa-padrão (% falsas): AD: Suf. 0, Rel. 0, Irr. 50, Con. 0; RF: Suf. 50, Rel. 0, Irr. 0, Con. 0; GB: Suf. 50, Rel. 0, Irr. 0, Con. 0; todos: Suf. 33, Rel. 0, Irr. 17, Con. 0.

Das afirmações verificáveis falsas do LLM, 71% eram autorizadas pela saída do SHAP (erro herdado da fonte) e 29% não eram (erro do narrador).

## Não inferioridade pareada por base (H1)

Margem fixada antes da execução: 10,0 pontos percentuais. Não inferior: o limite inferior unilateral de 95% da diferença média (LLM menos padrão), por reamostragem das bases, fica acima de menos a margem.

| Modelo | Bases | LLM (%) | Padrão (%) | Diferença média (p.p.) | Limite inferior 95% | IC 95% | Não inferior? | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---|---|---:|
| AD | 2 | 50 | 50 | 0.0 | -100.0 | -100.0 a 100.0 | não (indicativo: menos de 5 bases) | 0.500 |
| RF | 2 | 100 | 50 | 50.0 | 0.0 | 0.0 a 100.0 | sim (indicativo: menos de 5 bases) | 0.250 |
| GB | 2 | 50 | 50 | 0.0 | 0.0 | 0.0 a 0.0 | sim (indicativo: menos de 5 bases) | 0.250 |
| todos | 2 | 67 | 50 | 16.7 | -33.3 | -33.3 a 66.7 | não (indicativo: menos de 5 bases) | 0.500 |

## Cobertura da linguagem e fidelidade à saída do SHAP

| Tipo | Afirmações | % do total | Atributos reconhecidos (%) | Autorizadas pelo SHAP (%) | Falsas (%) | Falsas nos dados (%) | Atenuadas (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUF | 5 | 10 | 100 | 100 | 40 | 40 | 0 |
| REL | 20 | 38 | 100 | 100 | 10 | 0 | 20 |
| IRR | 6 | 12 | 100 | 50 | 17 | 100 | 0 |
| CON | 8 | 15 | 100 | 88 | 25 | 25 | 0 |
| ORD | 8 | 15 | 100 | 25 | -- | -- | 50 |
| DIR | 5 | 10 | 100 | 100 | -- | -- | 40 |
| OUTRA | 0 | 0 | -- | -- | -- | -- | -- |

ORD e DIR são conferidas só contra a saída do SHAP (fidelidade, como na literatura); OUTRA fica fora da linguagem. A cobertura indica quais extensões da linguagem (necessidade, contrafactuais com valores, ordem) mais fariam falta.

## Afirmações atenuadas e leitura branda da irrelevância

| Tipo | Atenuadas (n) | Falsas entre atenuadas (%) | Não atenuadas (n) | Falsas entre não atenuadas (%) |
|---|---:|---:|---:|---:|
| SUF | 0 | -- | 5 | 40 |
| REL | 4 | 25 | 16 | 6 |
| IRR | 0 | -- | 6 | 17 |
| CON | 0 | -- | 8 | 25 |

Irrelevância lida de forma estrita (o atributo não pertence a nenhuma explicação abdutiva): 17% falsas. Lida de forma branda (mudar só esse atributo não reverteria a decisão): 0% falsas.

Suficiências falsas do LLM são reparadas acrescentando, em média, 1,50 atributos.

## Por base

| Base | Modelo | Narrativas | Alguma (LLM) | Alguma (padrão) | Diferença | Herdado | Narrador |
|---|---|---:|---:|---:|---:|---:|---:|
| corral | AD | 1 | 100 | 0 | 100 | 100 | 100 |
| corral | RF | 1 | 100 | 0 | 100 | 100 | 0 |
| corral | GB | 1 | 0 | 0 | 0 | 0 | 0 |
| mux6 | AD | 1 | 0 | 100 | -100 | 0 | 0 |
| mux6 | RF | 1 | 100 | 100 | 0 | 0 | 100 |
| mux6 | GB | 1 | 100 | 100 | 0 | 100 | 0 |

## Exemplos

### corral, AD, instância 0 (classe: classe 0)

> A decisão de classificar o caso como "classe 0" foi influenciada principalmente pelo valor do atributo x2, que favoreceu essa classe. Os atributos x3 e x4 também contribuíram positivamente para a classificação em "classe 0", embora com menor intensidade. O valor de x1 teve um efeito negativo, mas foi menos significativo comparado aos outros fatores. A variável x6 não teve grande impacto na decisão. Se o valor do atributo x2 fosse diferente ou se x1 tivesse um valor que contrariasse sua influência, isso poderia mudar a classificação para "classe 1".

- SUF(x2, x3, x4) "a decisão se explica por x2 e x3 e x4": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- REL(x2) "x2 influenciou": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- REL(x3) "x3 contribuiu para a classificação em 'classe 0'": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- REL(x4) "x4 contribuiu para a classificação em 'classe 0'": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- REL(x1) "x1 teve um efeito negativo": **falsa**; nos dados, verdadeira; autorizada pelo SHAP
- IRR(x6) "x6 não teve grande impacto na decisão": verdadeira; nos dados, falsa; autorizada pelo SHAP
- CON(x2) "se x2 fosse diferente, a classificação poderia mudar para 'classe 1'": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- CON(x1) "se x1 tivesse um valor que contrariasse sua influência, isso poderia mudar a classificação para 'classe 1'": **falsa**; nos dados, falsa; não autorizada pelo SHAP
- ORD(x2) "x2 foi importante": não verificada (fora da linguagem); autorizada pelo SHAP
- ORD(x3) (atenuada) "x3 contribuiu positivamente para a classificação em 'classe 0', embora com menor intensidade": não verificada (fora da linguagem); não autorizada pelo SHAP
- ORD(x4) (atenuada) "x4 contribuiu positivamente para a classificação em 'classe 0', embora com menor intensidade": não verificada (fora da linguagem); não autorizada pelo SHAP
- ORD(x1) (atenuada) "x1 teve um efeito negativo, mas foi menos significativo comparado aos outros fatores": não verificada (fora da linguagem); não autorizada pelo SHAP

### mux6, GB, instância 0 (classe: classe 0)

> A decisão de classificar o caso como "classe 0" foi influenciada principalmente pelo valor do atributo x3, que teve um grande impacto positivo na probabilidade de ser "classe 0". Os atributos x1, x2 e x4 também contribuíram para essa decisão, embora com menor intensidade. O valor do atributo x6 não afetou a classificação neste caso, já que seu impacto foi mínimo. Se o valor do x3 fosse diferente ou se houvesse uma mudança significativa no valor de x1, isso poderia alterar a decisão do modelo.

- SUF(x3) "a decisão se explica por x3": **falsa**; nos dados, falsa; autorizada pelo SHAP
  - contraexemplo (dados de treino): mantidos x3 = 0; alterados x1 = 1 (em vez de valor 0); x5 = 1 (em vez de valor 0); classe classe 1.
- REL(x1) "os atributos x1, x2 e x4 contribuíram para essa decisão": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- REL(x2) "os atributos x1, x2 e x4 contribuíram para essa decisão": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- REL(x4) "os atributos x1, x2 e x4 contribuíram para essa decisão": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- IRR(x6) "o valor do atributo x6 não afetou a classificação neste caso": verdadeira; nos dados, falsa; autorizada pelo SHAP
- CON(x3) "se o valor do x3 fosse diferente, isso poderia alterar a decisão do modelo": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- CON(x1) "se houvesse uma mudança significativa no valor de x1, isso poderia alterar a decisão do modelo": **falsa**; nos dados, falsa; autorizada pelo SHAP
- DIR(x3) "x3 teve um grande impacto positivo na probabilidade de ser 'classe 0'": não verificada (fora da linguagem); autorizada pelo SHAP
- ORD(x1, x2, x4) (atenuada) "os atributos x1, x2 e x4 contribuíram para essa decisão, embora com menor intensidade": não verificada (fora da linguagem); não autorizada pelo SHAP

### corral, RF, instância 0 (classe: classe 0)

> A decisão de classificar o caso como "classe 0" foi influenciada principalmente pelo valor atribuído à variável x2, que teve um impacto positivo significativo na probabilidade estimada. As variáveis x4 e x3 também contribuíram positivamente, embora com menor intensidade. O valor de x1 teve um efeito negativo, mas foi menos influente do que o fator principal. Variáveis como x6 não pareceram ter grande impacto na decisão desta vez. Se x2 tivesse um valor diferente ou se x4 fosse zero, isso poderia mudar a classificação para "classe 1".

- SUF(x2) "a decisão se explica por x2 e contribuiu positivamente": **falsa**; nos dados, falsa; autorizada pelo SHAP
  - contraexemplo (dados de treino): mantidos x2 = 0; alterados x1 = 0 (em vez de valor 1); x3 = 1 (em vez de valor 0); x4 = 1 (em vez de valor 0); classe classe 1.
- REL(x4) (atenuada) "as variáveis x4 e x3 também contribuíram positivamente, embora com menor intensidade": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- REL(x3) (atenuada) "as variáveis x4 e x3 também contribuíram positivamente, embora com menor intensidade": verdadeira; nos dados, verdadeira; autorizada pelo SHAP
- REL(x1) (atenuada) "o valor de x1 teve um efeito negativo, mas foi menos influente do que o fator principal": **falsa**; nos dados, verdadeira; autorizada pelo SHAP
- IRR(x6) "variáveis como x6 não pareceram ter grande impacto na decisão desta vez": verdadeira; nos dados, falsa; não autorizada pelo SHAP
- CON(x2, x4) "se x2 tivesse um valor diferente ou se x4 fosse zero, isso poderia mudar a classificação para 'classe 1'": verdadeira; nos dados, verdadeira; autorizada pelo SHAP

## Leituras-padrão (autorização pela saída do SHAP)

- SUF: todos os atributos citados têm contribuição SHAP positiva (a favor da decisão).
- REL: o atributo tem contribuição SHAP não nula.
- IRR: a contribuição do atributo é desprezível (|phi_i| <= tau * max_j |phi_j|) ou é a de menor módulo entre os atributos fora dos três que mais favorecem a decisão (tau = 0.1).
- CON: algum atributo citado tem contribuição SHAP positiva.
- ORD: os atributos citados estão em ordem não crescente de |phi|; um atributo citado sozinho tem o maior |phi| ou o maior phi.
- DIR: o sinal de phi_i coincide com o sentido afirmado (a favor: phi_i > 0; contra: phi_i < 0).
