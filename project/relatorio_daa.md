# Previsão do Escalão Etário a partir do Inquérito Nacional de Saúde 2019

**Dados e Aprendizagem Automática 2026/27 — Trabalho Prático**

> Resultados obtidos com o notebook `project/projeto_daa.ipynb` (validação cruzada estratificada de 5 folds, métrica Macro F1 *out-of-fold*). Os números vêm de uma execução sem XGBoost, PyTorch e SHAP. Ao correr com estas bibliotecas, os valores podem mudar ligeiramente.

---

## 1. Introdução

O objetivo é prever o escalão etário (`AGE_RANGE`) de cada indivíduo entre **16 classes**, dos 15–17 anos aos 85+, a partir de 237 variáveis do INS 2019. Essas variáveis cobrem a demografia, a situação de trabalho, a saúde, as doenças crónicas, a autonomia e os estilos de vida. O trabalho tem também dois objetivos de análise: identificar **anomalias epidemiológicas**, ou seja, pessoas cujo perfil parece mais velho ou mais novo do que a idade real, e **caracterizar cada escalão**.

A métrica é a **Macro F1**: todas as classes pesam o mesmo, independentemente do número de indivíduos que têm.

| | Treino | Teste |
|---|---|---|
| Linhas | 10 962 | 3 655 |
| Colunas | 237 (+ alvo) | 236 |

---

## 2. Compreensão dos dados

### 2.1 Distribuição do alvo
As classes estão desequilibradas. Os escalões 15–17 e 18–19 têm cerca de 5× menos indivíduos do que os 60–64. Com Macro F1, errar sistematicamente as classes pequenas custa tanto como errar as grandes, por isso todos os modelos usam pesos por classe (`class_weight='balanced'`).

### 2.2 Codificação das respostas
Quase todas as colunas vêm como **texto** (por exemplo, `"Sim"`, `"Nenhuma dificuldade"`). Nas colunas numéricas aparecem três códigos especiais:

| Código | Significado | Tratamento |
|---|---|---|
| `Não aplicável` | A pergunta **não foi feita** (regra de salto) | Valor próprio (-1 ou categoria) |
| `Sem resposta` | O inquirido não respondeu | Valor próprio (-2 ou categoria) |
| `>= 25`, `>=7` | Valor truncado no topo | Convertido para o número |

`Não aplicável` **não é um valor em falta**: diz por que caminho do questionário a pessoa passou, e isso depende muito da idade.

### 2.3 Regras de salto do questionário (achado principal)
Para cada variável, procurámos os escalões em que a resposta é **sempre** `Não aplicável`, e encontrámos 48 variáveis nesta situação. As mais relevantes são:

| Módulo | Variáveis | Só perguntado a… |
|---|---|---|
| Cuidados pessoais e tarefas domésticas | `PC1A–E`, `HA1A–G`, `IN15` | **≥ 55 anos** |
| Escolaridade/trabalho | `IN2`, `IN3` | ≥ 20 anos |
| Atividade profissional | `JOBSTAT_COD`, `JOBISCO_COD`, `LOCNACE_COD`, `FT_PT` | 18–74 anos |
| Módulo NT | `NT1`, `NT2`, `IN20`–`IN28` | < 60 anos (parcial) |
| Tabaco | `SK2`, `IN49`, `IN4` | < 80 anos |

**Consequência:** `PC1A == 'Não aplicável'` separa **perfeitamente** as classes 0–8 (< 55) das classes 9–15 (≥ 55). O modelo nunca confunde estes dois blocos. Isto é um **artefacto do desenho do inquérito**, não uma característica de saúde. É informação legítima para a competição, mas deve ser tido em conta ao interpretar o modelo.

---

## 3. Alterações ao dataset

Todas as transformações estão registadas na tabela `CHANGES` do notebook (secção 2.5):

| # | Etapa | Alteração | Colunas | Motivo |
|---|---|---|---|---|
| 1 | Limpeza | Remoção de identificadores | `PID`, `HHID` | Não descrevem o indivíduo; risco de overfitting (cada `HHID` é único) |
| 2 | Limpeza | Remoção de colunas constantes | `PROXY`, `COUNTRY` | Um único valor em todo o dataset |
| 3 | Tipos | Conversão texto → número | 41 colunas (ex.: `BM1`, `BM2`, `PE*`) | > 95% dos valores são numéricos |
| 4 | Tipos | `Não aplicável` → -1, `Sem resposta` → -2 | colunas numéricas | Manter a informação do salto como valor próprio |
| 5 | Tipos | `>= 25` → 25 (e análogos) | colunas truncadas | Valor truncado no topo |
| 6 | Tipos | Texto → categoria (NaN → `"NA"`) | 191 colunas | Variáveis qualitativas do inquérito |
| 7 | Feature eng. | `BMI` = peso / altura² | `BM1`, `BM2` → `BMI` | O IMC é mais interpretável do que peso e altura separados |
| 8 | Feature eng. | Nº de doenças crónicas | `CD1A–P` → `N_CHRONIC` | Resumo da carga de doença |
| 9 | Feature eng. | Nº de limitações em tarefas | `PC1*`, `HA1*` → `N_LIMIT` | Resumo da perda de autonomia |
| 10 | Feature eng. | Nº de respostas `Não aplicável` | todas → `N_NAO_APLICAVEL` | Captura o percurso no questionário |
| 11 | Feature eng. | Indicador do módulo ≥ 55 anos | `PC1A` → `MODULO_55PLUS` | Fronteira exata dos 55 anos |
| 12 | Codificação | `OrdinalEncoder` + NaN → -3 | categóricas | Decision Tree, Random Forest e Bagging só aceitam números |
| 13 | Codificação | One-hot (`min_frequency=10`) + `StandardScaler` | todas → 1031 colunas | Logistic Regression e MLP precisam de escalas comparáveis |
| 14 | Codificação | Categorias nativas | categóricas | Boosting (HGB, XGBoost) trata categorias e NaN internamente |
| 15 | Desequilíbrio | `class_weight='balanced'` / `sample_weight` | `AGE_RANGE` | As classes pequenas pesam o mesmo na Macro F1 |

---

## 4. Relações entre as variáveis e a idade

Medimos a força da relação de cada variável com o escalão com a **informação mútua** (que capta relações não lineares e funciona com categorias) e com a **correlação de Spearman** (relações monótonas nas variáveis numéricas).

**Principais relações encontradas:**

1. **Ciclo de vida profissional (`MAINSTAT`):** o emprego passa de 0.5% aos 15–17 anos para um máximo de 87% aos 35–39. A reforma sobe de 7% (55–59) para 31% (60–64) e 72% (65–69), e chega a 90% a partir dos 70.
2. **Estado civil (`MARSTALEGAL`):** a viuvez é quase nula até aos 50 e sobe para 16% (65–69), 40% (75–79) e 64% (85+).
3. **Escolaridade (`HATLEVEL`):** há um efeito geracional. O ensino superior chega aos 34% nos 25–29 e desce para 5% nos 85+. Sem qualquer escolaridade estão menos de 1% até aos 44 anos e 47% nos 85+.
4. **Saúde:** o número médio de doenças crónicas sobe de forma monótona, de 0.5 para 4.4. A hipertensão passa de cerca de 1% para 65%, a diabetes de 0% para 29% e a saúde má ou muito má de 0.5% para 43%.
5. **Autonomia:** a dificuldade em andar sobe de 1% para 43% e a limitação severa de 0.5% para 32% (85+).
6. **Estilos de vida:**
   - o tabaco diário tem um pico entre os 25 e os 45 anos (cerca de 24%) e desce para 1% depois dos 80;
   - o álcool diário cresce até aos 65–69 (28%);
   - a fruta diária aumenta com a idade (de 48% para 75%).
7. **Rendimento:** os mais jovens estão mais vezes no 1.º quintil (64% nos 15–17), porque vivem em agregados com estudantes ou sem rendimentos próprios.
8. **IMC:** sobe até aos 65–69 (27.5) e desce nos mais velhos (25.7 nos 85+).
9. **Composição do agregado** (`HH_ACT`, `HH_INACT`, `HHTYPE`, `PARTNERS`, nº de crianças): viver com os pais, com filhos pequenos, sozinho ou em casal reformado é muito típico de cada idade.

**Redundância:** as variáveis do mesmo módulo (por exemplo `PC1A–E` e `HA1A–G`) estão muito correlacionadas entre si, por isso a importância divide-se entre elas.

---

## 5. Features mais relevantes

Combinámos três métodos (aula 7), porque cada um tem enviesamentos próprios:

- **Informação mútua:** relação direta com a idade, independente do modelo.
- **MDI (Mean Decrease in Impurity) da Random Forest:** específica do modelo e calculada no treino. Favorece variáveis com muitas categorias.
- **Permutation importance do HistGradientBoosting:** independente do modelo e calculada em validação. Mede a queda da Macro F1 quando se baralha cada variável.

| Rank | Variável | Significado | MI | MDI | Permutação |
|---|---|---|---|---|---|
| 1 | `MAINSTAT` | Condição perante o trabalho | 15 | **1** | 2 |
| 2 | `HH_ACT` | Nº de ativos no agregado | 18 | 6 | 7 |
| 3 | `MARSTALEGAL` | Estado civil | 23 | 4 | 6 |
| 4 | `HATLEVEL` | Escolaridade | 26 | 3 | 4 |
| 5 | `PC1A` | Cuidados pessoais (módulo ≥ 55) | 13 | 19 | **1** |
| 6 | `IN15` | Filtro do módulo ≥ 55 | 12 | 22 | 11 |
| 7 | `HHTYPE` | Tipo de agregado | 24 | 24 | 5 |
| 8 | `JOBSTAT_COD` | Situação na profissão | 16 | 10 | 28 |
| 9 | `HH_INACT` | Nº de inativos no agregado | 51 | **2** | 3 |
| 10 | `HA1C` | Tarefas domésticas (módulo ≥ 55) | **2** | 30 | 24 |

**Conclusão:** a idade é prevista sobretudo pela **idade "social"**, isto é, pelo trabalho, pelo estado civil, pela escolaridade e pela composição do agregado, e pelo **percurso no questionário**. As variáveis de saúde estão relacionadas com a idade, mas pesam menos, porque variam muito entre pessoas da mesma idade.

---

## 6. Modelos experimentados

Todos os modelos foram avaliados com o mesmo protocolo: **StratifiedKFold com 5 folds** e Macro F1 calculada sobre as previsões *out-of-fold*.

| Modelo | Aula | Configuração |
|---|---|---|
| Decision Tree | PL5 | `GridSearchCV` → `max_depth=10`, `min_samples_leaf=20` |
| Logistic Regression | PL4 | One-hot + normalização, `C=0.05` |
| Random Forest | PL6–7 | 400 árvores, `min_samples_leaf=3` |
| Bagging (Decision Trees) | PL6 | 60 árvores, `max_samples=0.8` |
| HistGradientBoosting | PL6 | `learning_rate=0.06`, até 600 iterações, early stopping |
| XGBoost | PL6 | 700 árvores, `max_depth=6` *(não executado nesta corrida)* |
| MLP | PL8 | 2 camadas ocultas (256, 64), Adam, early stopping |
| Soft Voting | PL6 | Média de probabilidades dos 3 melhores modelos |
| **Stacking** | PL6 | `LogisticRegression` sobre as probabilidades de todos os modelos |

O MLP segue a estrutura da aula 8, com duas melhorias tiradas das conclusões dos modelos M1–M5:
- **sem `Softmax` na última camada**, porque a `CrossEntropyLoss` já o aplica;
- **todas as variáveis normalizadas**, porque a normalização parcial do M3 prejudicou os resultados.

---

## 7. Resultados

### 7.1 Comparação global

| # | Modelo | Família | Macro F1 | ± folds | Accuracy | Erro médio (classes) | Tempo |
|---|---|---|---|---|---|---|---|
| 🥇 | **Stacking (LR)** | Ensemble de modelos | **0.474** | — | 0.482 | **0.71** | — |
| 2 | Voting (top-3) | Ensemble de modelos | 0.463 | — | 0.473 | 0.73 | — |
| 3 | HistGradientBoosting | Boosting | 0.457 | 0.007 | 0.466 | 0.76 | 127 s |
| 4 | Bagging (DT) | Bagging | 0.440 | 0.013 | 0.447 | 0.82 | 31 s |
| 5 | Logistic Regression | Linear | 0.428 | 0.009 | 0.436 | 0.82 | 38 s |
| 6 | Random Forest | Bagging | 0.426 | 0.008 | 0.443 | 0.82 | 26 s |
| 7 | MLP | Rede neuronal | 0.419 | 0.007 | 0.426 | 0.82 | 60 s |
| 8 | **Decision Tree** | Árvore | **0.395** | 0.017 | 0.390 | 0.98 | 1 s |

Com pesos por classe otimizados para a Macro F1, o **Stacking chega a 0.486**. Este é o modelo usado na submissão.

### 7.2 Melhor modelo: Stacking
- É o melhor em **10 dos 16 escalões** e **nunca é o pior**.
- Junta modelos de famílias diferentes (árvores, linear, rede neuronal) que erram em casos diferentes. O meta-modelo aprende em que modelo confiar para cada classe.
- Tem o menor erro médio: 0.71 classes, ou seja, cerca de 3.5 anos.

### 7.3 Melhor modelo individual: HistGradientBoosting
- É o melhor modelo isolado (0.457) e o mais estável entre folds (±0.007).
- Ganha em 4 escalões: 15–17 (0.85), 20–24 (0.59), 45–49 e 75–79.
- Trata nativamente as variáveis categóricas e os `Não aplicável`, e capta interações entre variáveis.

### 7.4 Pior modelo: Decision Tree
- É o pior em **11 dos 16 escalões** e tem a maior variância entre folds (±0.017).
- Uma única árvore com profundidade 10 não consegue separar 16 classes tão próximas: ou é demasiado simples, ou sobreajusta.
- Mesmo assim, é a base dos modelos melhores. O Bagging de árvores sobe para 0.440 e o Boosting para 0.457, o que mostra o valor dos ensembles.

### 7.5 Outros resultados
- **Logistic Regression (0.428):** é surpreendentemente competitiva, porque grande parte da informação está em variáveis categóricas com relações quase monótonas com a idade. É a pior só nos 70–74.
- **Random Forest (0.426):** fica abaixo do Bagging simples. A escolha aleatória de variáveis (`sqrt` de 237) deixa muitas vezes de fora as poucas variáveis decisivas.
- **MLP (0.419):** fica abaixo dos ensembles de árvores, o que é habitual em dados tabulares com muitas variáveis categóricas. É o pior nos 15–17.

### 7.6 Melhores e piores modelos por escalão

| Escalão | Melhor modelo | F1 | Pior modelo | F1 | F1 médio |
|---|---|---|---|---|---|
| 15–17 | HistGradientBoosting | **0.848** | MLP | 0.722 | 0.800 |
| 18–19 | Stacking | 0.413 | HistGradientBoosting | 0.288 | 0.352 |
| 20–24 | HistGradientBoosting | 0.585 | Random Forest | 0.432 | 0.517 |
| 25–29 | Stacking | 0.400 | Decision Tree | 0.300 | 0.347 |
| 30–34 | Stacking | 0.280 | Random Forest | 0.204 | **0.244** |
| 35–39 | Random Forest | 0.360 | Decision Tree | 0.299 | 0.339 |
| 40–44 | Stacking | 0.348 | Decision Tree | 0.244 | 0.316 |
| 45–49 | HistGradientBoosting | 0.410 | Decision Tree | 0.315 | 0.360 |
| 50–54 | Stacking | 0.658 | Decision Tree | 0.522 | 0.611 |
| 55–59 | Stacking | **0.684** | Decision Tree | 0.601 | 0.654 |
| 60–64 | Stacking | 0.657 | Decision Tree | 0.556 | 0.611 |
| 65–69 | Stacking | 0.575 | Decision Tree | 0.423 | 0.491 |
| 70–74 | Bagging (DT) | 0.396 | Logistic Regression | 0.333 | 0.363 |
| 75–79 | HistGradientBoosting | 0.294 | Decision Tree | 0.216 | **0.263** |
| 80–84 | Stacking | 0.320 | Decision Tree | 0.251 | 0.293 |
| 85+ | Stacking | 0.483 | Decision Tree | 0.337 | 0.441 |

**Escalões mais fáceis:**
- **15–17:** quase todos são estudantes, sem emprego e com o 3.º ciclo como escolaridade máxima.
- **50–64:** estão junto à fronteira exata dos 55 anos (regra de salto) e à transição para a reforma.

**Escalões mais difíceis:**
- **30–34, 40–44, 75–79 e 80–84:** ficam no meio de cada bloco, onde nada muda de forma marcada em 5 anos (emprego estável, ou reforma já consolidada).
- **18–19:** é uma classe pequena e muito parecida com os 20–24.

### 7.7 Porque é que a Macro F1 fica perto de 0.5
Um modelo ao acaso teria uma Macro F1 de cerca de 0.06 (1/16). O modelo raramente erra por muito:

| Critério | Resultado |
|---|---|
| Escalão exato | 49% |
| Erro de 1 escalão no máximo (±5 anos) | **87%** |
| Erro de 2 escalões no máximo (±10 anos) | **97%** |

Se as classes forem agrupadas, o mesmo modelo tem:

| Agrupamento | Macro F1 |
|---|---|
| 16 escalões de 5 anos | 0.49 |
| 8 grupos de cerca de 10 anos | 0.71 |
| 4 grupos (jovem, adulto, sénior, idoso) | 0.85 |

O modelo identifica bem a **fase da vida**. A dificuldade está em distinguir 5 anos dentro da mesma fase, onde os perfis se sobrepõem.

---

## 8. Anomalias epidemiológicas

**Método:**
1. Calculamos a idade prevista esperada a partir das probabilidades *out-of-fold* do Stacking: $\hat{a} = \sum_k p_k \cdot k$. Como são OOF, o modelo nunca viu a idade real dessa pessoa.
2. O desvio bruto $\hat{a} - a$ sofre de **regressão para a média**: os mais novos de cada bloco são puxados para cima e os mais velhos para baixo. Por isso, o desvio é **padronizado dentro de cada escalão real** (z-score).
3. Uma anomalia é um caso com $|z| > 2$.

**Resultado:** **217 perfis envelhecidos** e **258 perfis atipicamente jovens**, num total de 10 962.

| Indicador (diferença em pp face aos pares da mesma idade) | Envelhecidos | Jovens |
|---|---|---|
| Reformado | **+17.5** | −18.8 |
| Dificuldade em andar | **+16.1** | −5.8 |
| Severamente limitado | **+15.9** | −3.8 |
| Saúde má/muito má | **+15.6** | −6.1 |
| Viúvo | +12.5 | −11.5 |
| Sem escolaridade | +12.2 | −5.1 |
| Hipertensão | +8.1 | **−11.8** |
| Diabetes | +7.8 | −3.0 |
| Colesterol | +3.3 | **−10.6** |
| Empregado | **−14.0** | +3.0 |
| 1.º quintil de rendimento | +1.6 | **+14.8** |
| Fruta diária | −1.1 | −12.0 |
| Fuma diariamente | −0.7 | +6.1 |

**Interpretação:**
- **Perfis envelhecidos:** pessoas com reforma antecipada, incapacidade, limitações de mobilidade, pior saúde e menos escolaridade. Correspondem a um **envelhecimento biológico e social precoce**.
- **Perfis jovens:** menos doença cardiovascular e metabólica, mas mais pobreza, mais tabaco e menos fruta. Um perfil "jovem" **nem sempre significa boa saúde**: pode refletir uma situação socioeconómica típica de idades mais novas, ou ainda não ter diagnósticos registados.
- Alguns casos extremos podem ser **erros de registo** da idade ou respostas *proxy*.

---

## 9. Caracterização dos escalões

| Fase | Escalões | Perfil |
|---|---|---|
| **Estudantes** | 15–19 | Sem emprego, 1.º quintil de rendimento (55–64%), quase sem doenças crónicas (≤ 0.7), pouco tabaco e álcool, IMC ≈ 22.5 |
| **Entrada na vida ativa** | 20–29 | O emprego sobe de 46% para 75%, o ensino superior chega ao máximo (34%), o tabaco diário fica perto de 20–24% |
| **Adultos ativos** | 30–49 | Emprego entre 83% e 87%, doenças crónicas a subir (1.0 → 1.6), hipertensão e colesterol começam a aparecer (8–17%), o IMC passa os 25 |
| **Pré-reforma** | 50–59 | O emprego desce para 68%, a hipertensão vai a 28–34%, a diabetes duplica (8% → 12%), o álcool diário chega a 25% |
| **Transição para a reforma** | 60–69 | A reforma passa de 31% para 72%, a hipertensão de 45% para 52%, o colesterol fica perto de 48%, há mais viuvez (12–16%) |
| **Idosos** | 70–79 | A reforma passa os 88%, a viuvez vai a 27–40%, a saúde má a 24–35%, as doenças crónicas a 3.8–4.3, o tabaco fica perto de 3% |
| **Muito idosos** | 80+ | Viuvez entre 53% e 64%, cerca de 45% sem escolaridade, dificuldade em andar entre 30% e 43%, limitação severa entre 23% e 32%, IMC a descer |

---

## 10. Conclusões e trabalho futuro

**Conclusões principais:**
1. **Melhor modelo: Stacking**, com Macro F1 de 0.474, ou **0.486** com pesos por classe. Combinar modelos diversos é a técnica que mais ajuda.
2. **Melhor modelo individual:** o HistGradientBoosting (0.457).
3. **Pior modelo:** a Decision Tree (0.395). Os ensembles corrigem a sua variância.
4. **A idade "social" prevê melhor do que a "biológica":** as variáveis mais importantes são o trabalho, o estado civil, a escolaridade e o agregado familiar. A saúde é um sinal mais fraco.
5. **As regras de salto do questionário** dão fronteiras exatas aos 55 anos (e aproximadas aos 20, 60, 75 e 80). Ajudam o modelo, mas são um artefacto do inquérito.
6. **Os erros quase sempre caem num escalão vizinho:** 87% das previsões ficam a ±5 anos da idade real.
7. **Anomalias:** 217 perfis envelhecidos, associados a incapacidade, reforma precoce e pior saúde, e 258 perfis jovens, que nem sempre correspondem a melhor saúde.

**Trabalho futuro:**
- Afinar o XGBoost, o LightGBM e o CatBoost com `RandomizedSearchCV` ou Optuna.
- Aproveitar a **ordem** das classes: juntar ao ensemble um modelo de regressão ordinal.
- Criar mais features ao nível do módulo do questionário, como contagens de `Não aplicável` por módulo.
- Usar SHAP para explicar individualmente cada anomalia.
