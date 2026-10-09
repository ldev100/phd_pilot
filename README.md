# Narrativas de explicação verificáveis

Estudo preliminar que acompanha o projeto de pesquisa *Narrativas de Explicação
Verificáveis: Checagem Formal e Certificada de Explicações em Linguagem Natural para
Modelos de Aprendizado de Máquina*.

## Objetivo

Explicações de decisões automatizadas chegam cada vez mais às pessoas na forma de texto,
muitas vezes escrito por um modelo de linguagem a partir da saída de ferramentas como o
SHAP. Uma narrativa desse tipo faz **afirmações** sobre o modelo: "a decisão se explica
por A, B e C", "A influenciou a decisão", "se A fosse diferente, a decisão poderia
mudar", "Z não influenciou o resultado".

Este projeto pergunta: **se uma narrativa adotar as leituras usuais da saída do SHAP, sem
nenhuma alucinação, com que frequência suas afirmações serão formalmente falsas em relação
ao próprio modelo?** E quanto custa, em tamanho, uma narrativa verdadeira por construção?

## Linguagem de afirmações

Cada afirmação tem semântica formal relativa a uma instância (v, c) e a um domínio de
entradas C (espaço completo ou dados de treino):

| Afirmação | Exemplo em linguagem natural | Semântica |
|---|---|---|
| Suficiência `SUF(T)` | "a decisão se explica por A, B e C" | T é uma explicação abdutiva fraca: todo x em C com x_T = v_T recebe a classe c |
| Relevância `REL(i)` | "A influenciou a decisão" | i pertence a alguma explicação abdutiva (AXp) |
| Irrelevância `IRR(i)` | "Z não influenciou o resultado" | i não pertence a nenhuma AXp |
| Contraste `CON(T)` | "se A fosse diferente, a decisão poderia mudar" | existe x em C que só difere de v em T e recebe outra classe |

A **narrativa-padrão derivada do SHAP** afirma a suficiência dos (até) três atributos de
maior contribuição a favor da decisão, a relevância e o contraste do primeiro deles e a
irrelevância do atributo de contribuição mais próxima de zero entre os demais, isto é, lê
uma contribuição desprezível como ausência de influência. A **narrativa verificada**
afirma a suficiência de uma explicação abdutiva mínima, a relevância de um atributo
relevante, o contraste de uma explicação contrastiva mínima e a irrelevância de um
atributo formalmente irrelevante (se houver), com empates resolvidos pela maior soma das
contribuições SHAP positivas. Todas as suas afirmações são verdadeiras por construção.

## Configuração

- 9 bases com espaço de entrada pequeno o bastante para verificação exata por enumeração:
  seis conceitos lógicos clássicos e três bases reais embutidas no scikit-learn, reduzidas a 4 a 8 variáveis e discretizadas.
- 3 modelos por base: árvore de decisão (profundidade 5), *random forest* (50 árvores,
  profundidade 5) e *gradient boosting* (50 árvores, profundidade 3).
- Até 50 instâncias de teste por base e modelo: **1.167 narrativas** no total.
- SHAP: explicador intervencional para árvores, 100 exemplos de treino como referência,
  probabilidade da classe predita.
- Verificação exata: tabela de suficiência para todos os 2^m subconjuntos de variáveis,
  por transformada de soma sobre superconjuntos, no espaço completo e nos dados de treino.

## Como reproduzir

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_claims.py 
python -m pytest -q                       
python run_pilot.py                       
python run_pilot.py --quick               
```

Saídas em `results/`: CSVs por instância, por base, por grupo de bases e por modelo.

## Extensão: narrativas redigidas por um LLM aberto

O script `run_llm_narratives.py` repete o experimento 1 com um narrador de verdade, um LLM
aberto executado no próprio computador. Ele usa as mesmas instâncias e os mesmos valores
SHAP do piloto, o que permite a comparação pareada com a narrativa-padrão prevista na
hipótese H1 do projeto. Para cada instância, são feitas duas chamadas ao LLM:

1. **Narração.** O LLM recebe o contexto da tarefa, os valores do caso, a decisão e as
   contribuições SHAP e escreve uma explicação curta para uma pessoa leiga. Há dois estilos
   de instrução: `guiado`, que pede os fatores que explicam a decisão, o que mais pesou,
   algum fator sem influência e o que mudaria a decisão, e `livre`, que pede apenas a
   explicação. Nas bases sintéticas, os atributos aparecem como x1, ..., xm, porque os
   nomes originais (como "Irrelevante") e a descrição dos conceitos revelariam a resposta.
2. **Extração.** O LLM lista em JSON as afirmações do texto, nos tipos SUF, REL, IRR e CON
   da linguagem do piloto, mais ORD (ordem de importância), DIR (sentido de uma
   contribuição) e OUTRA. A saída é restrita por um esquema JSON cujos nomes de atributo
   só podem ser os do modelo, o que elimina erros de grafia.

Cada afirmação SUF, REL, IRR e CON é então decidida de forma exata, no espaço completo e
nos dados de treino, e classificada como **autorizada ou não pela saída do SHAP**, segundo
leituras-padrão fixadas antes da execução (`LEITURAS` em `verif/llm.py`). Afirmações
autorizadas e falsas medem o **erro herdado da fonte**; afirmações não autorizadas e falsas
medem o **erro do narrador**. As leituras-padrão generalizam as da narrativa-padrão:

| Tipo | A saída do SHAP autoriza a afirmação se... |
|---|---|
| SUF | todos os atributos citados têm contribuição positiva (a favor da decisão) |
| REL | o atributo tem contribuição não nula |
| IRR | a contribuição é desprezível (no máximo 10% da maior, `--tau`) ou é a menor fora dos três atributos que mais favorecem a decisão |
| CON | algum atributo citado tem contribuição positiva |
| ORD | os atributos citados estão em ordem não crescente de contribuição absoluta |
| DIR | o sinal da contribuição coincide com o sentido afirmado |

ORD e DIR são conferidas só contra a saída do SHAP, como nas medidas de fidelidade da
literatura. Para a irrelevância, o script também informa uma leitura branda ("mudar só esse
atributo não reverteria a decisão"), mais fraca que a estrita ("o atributo não pertence a
nenhuma explicação abdutiva"). As afirmações com ressalva ("principalmente", "em parte")
são marcadas como atenuadas e analisadas à parte.

### Como executar

1. Instale o [Ollama](https://ollama.com/download) e baixe um
   modelo, por exemplo `ollama pull qwen2.5:7b`. Em modelos que raciocinam antes de responder, como o `qwen3`, use `--think off`.
2. Confira o encadeamento sem LLM. Esse modo reproduz os números da
   narrativa-padrão:
   ```bash
   python run_llm_narratives.py --backend mock --quick
   ```
3. Faça um teste curto com o LLM e leia as narrativas em `resumo_llm.md`:
   ```bash
   python run_llm_narratives.py --model qwen2.5:7b --max-cases 6
   ```
4. Rode o experimento.
   ```bash
   python run_llm_narratives.py --model qwen2.5:7b --instances 10   # 270 narrativas
   python run_llm_narratives.py --model qwen2.5:7b                  # 1.167 narrativas
   python run_llm_narratives.py --model qwen2.5:7b --style livre
   ```
5. Compare narradores e estilos já executados:
   ```bash
   python run_llm_narratives.py --combine results_llm
   ```

### Saídas

Em `results_llm/<modelo>/<estilo>/`:

- `resumo_llm.md`: tabelas comentadas e exemplos com vereditos e contraexemplos;
- `tabela_llm.tex`: tabela para o texto, LLM contra narrativa-padrão;
- `afirmacoes.csv`: uma linha por afirmação, com tipo,
  atributos, veredito nos dois domínios e autorização;
- `por_narrativa.csv`, `por_modelo_llm.csv`, `por_tipo.csv` e `por_base_llm.csv`: agregados;
- `nao_inferioridade.csv`: teste pareado por base da hipótese H1. A margem é de 10 pontos
  percentuais e pode ser alterada com `--margin`, sempre antes da execução;
- `narrativas.jsonl`: registro completo de cada caso (entrada, texto, extração bruta,
  vereditos e contraexemplos);
- `anotacao.csv`: amostra estratificada para dois anotadores conferirem a extração, aberta
  diretamente em planilhas configuradas em português;
- `instrucoes.json` e `execucao.json`: instruções enviadas ao LLM, versão e *digest* do
  modelo, opções e ambiente.

O arquivo `cache.jsonl` guarda as respostas brutas para retomar a execução e não é
versionado.

## Estrutura

```
verif/datasets.py      bases de dados (geradas localmente ou embutidas no scikit-learn)
verif/oracle.py        tabelas exatas de suficiência relativas a um domínio finito
verif/claims.py        linguagem de afirmações, semântica formal e narrativas
verif/shap_tool.py     ferramenta SHAP orientada à classe predita
verif/experiment.py    configuração compartilhada: semente, modelos e instâncias
verif/llm.py           narração e extração por LLM, autorização pelo SHAP, clientes HTTP
run_pilot.py           experimentos do piloto, agregação e tabelas
run_llm_narratives.py  extensão com LLM aberto
tests/                 testes automatizados (com um servidor falso que imita o Ollama)
```

## Limitações

- A verificação por enumeração só é viável em espaços de entrada pequenos. No projeto,
  ela será substituída por raciocínio SAT/MaxSAT sobre codificações do modelo.
- Os resultados do piloto não usam um modelo de linguagem: a narrativa-padrão é gerada por
  modelos de frase e representa as leituras usuais da saída do SHAP. A extensão com LLM
  aberto, descrita acima, mede o erro do próprio narrador. Nela, a extração das afirmações
  também é feita por um LLM e precisa ser conferida por anotadores (`anotacao.csv`) antes
  de qualquer conclusão.
- A leitura formal de cada frase é uma escolha explícita (por exemplo, "se explica por"
  como suficiência e "não influenciou" como irrelevância). Tornar essa escolha explícita e
  verificável é justamente o objetivo da linguagem de afirmações proposta.
- A seleção de variáveis e os cortes de quantis das bases reais usam todos os dados, como no
  piloto anterior; o efeito sobre as conclusões é pequeno, mas será evitado no projeto.
- Os conceitos MONK têm variáveis com mais de dois valores, e o ruído do MONK3 é aplicado
  sobre os 432 pontos possíveis.
