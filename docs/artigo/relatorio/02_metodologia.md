# 2. Metodologia 

Detecção de formigueiros em ortofotos aéreas via segmentação semântica com U-Net.

> Este documento descreve **cada técnica utilizada**, agrupada por etapa do pipeline (preparação → treinamento → pós-processamento → avaliação). Para o histórico cronológico de cada experimento e suas métricas, ver [EXPERIMENTS.md](EXPERIMENTS.md).
>
> **Padrão de cada técnica:** nome técnico + explicação acessível em pt-BR + motivação no contexto deste projeto.

---

## Índice

1. [Visão Geral do Problema](#1-visão-geral-do-problema)
2. [Preparação dos Dados](#2-preparação-dos-dados)
3. [Augmentação de Dados](#3-augmentação-de-dados)
4. [Arquitetura do Modelo](#4-arquitetura-do-modelo)
5. [Funções de Custo (Loss)](#5-funções-de-custo-loss)
6. [Otimização do Treinamento](#6-otimização-do-treinamento)
7. [Pós-processamento das Predições](#7-pós-processamento-das-predições)
8. [Métricas de Avaliação](#8-métricas-de-avaliação)

---

## 1. Visão Geral do Problema

**Tarefa:** dado um recorte (tile) de uma ortofoto aérea de talhão agrícola, identificar os pixels que pertencem a formigueiros (cupinzeiros/saúveiros).

**Características do problema que motivam todas as decisões a seguir:**

| Característica | Implicação |
|---|---|
| **Desbalanceamento extremo** — formigueiros ocupam <1% dos pixels | Loss de classificação simples (Cross-Entropy) tende a "prever sempre fundo" e ainda assim acertar 99% |
| **Baixa área da classe positiva** — 2.406 das 9.862 imagens (~24%) contêm formigueiro, mas a classe ocupa <1% dos pixels | Sem augmentação, o modelo decora os padrões positivos disponíveis |
| **Solo avermelhado parecido com formigueiro** | Confunde o modelo; aumenta falsos positivos |
| **Bordas dos formigueiros são mal definidas** | Métricas de IoU (sobreposição pixel-a-pixel) ficam baixas mesmo com detecção correta |
| **Tiles com 70-95% de borda branca (área não rotulada)** | Treinar nesses tiles desperdiça gradiente — quase nenhum sinal de aprendizagem |

---

## 2. Preparação dos Dados

### 2.1 Estrutura do dataset

**O que é:** o conjunto de dados está organizado em pares RGB+máscara, separados em treino e validação.

```
data/
├ training/
│   ├ rgb/rgb/          ← 9.862 imagens RGB (ortofotos recortadas em tiles)
│   └ labels/labels/    ← 9.862 máscaras correspondentes (PNG com cores)
└ validation/
    ├ rgb/rgb/          ← 2.466 imagens RGB
    └ labels/labels/    ← 2.466 máscaras
```

**Pareamento:** [app/infrastructure/segmentation_dataset.py](app/infrastructure/segmentation_dataset.py) faz o match exato pelo nome do arquivo (sem extensão), garantindo que cada RGB tenha sua máscara correspondente.

---

### 2.2 Decodificação RGB das máscaras

**O que é:** as máscaras vêm como imagens RGB coloridas, não como classes numéricas. Cada cor representa uma classe diferente.

**Como funciona (simples):** cada pixel da máscara é "lido" pela cor:
- **Vermelho** (R>150, G<100, B<100) → classe **1** (formigueiro)
- **Preto** (R,G,B < 50) → classe **0** (fundo / solo)
- **Branco** (R,G,B > 200) → classe **255** (ignorar — região não rotulada manualmente)

**Por que aplicado:** as anotações manuais foram feitas pintando regiões em vermelho sobre as ortofotos. Manter o formato de cor permite revisão visual fácil pelo anotador. A classe "ignorar" é fundamental: pixels não rotulados (bordas onde o anotador não pintou) são **excluídos do cálculo da loss e das métricas**, evitando que o modelo aprenda padrões a partir de áreas duvidosas.

**Onde:** [app/infrastructure/segmentation_dataset.py:296-307](app/infrastructure/segmentation_dataset.py#L296-L307) e [run_evaluate.py:56-73](run_evaluate.py#L56-L73).

---

### 2.3 Filtro de tiles dominados por área ignorada

**O que é:** descartar do conjunto de treino tiles cujas máscaras tenham mais de 70% de pixels brancos (área não rotulada).

**Como funciona (simples):** ao construir o dataset, o código abre cada máscara e calcula a fração de pixels brancos. Se mais de 70% da imagem é "ignorar", a tile é descartada. **O conjunto de validação não é filtrado** — assim métricas continuam comparáveis.

**Por que aplicado:** muitas tiles na borda do talhão estão 70-95% fora da área anotada. Treinar com elas:
- Desperdiça tempo de GPU (cada tile gasta um forward+backward sem aprender quase nada)
- Reduz o sinal médio de gradiente por batch (a maior parte dos pixels é "ignorar" → não contribui para a loss)

**Resultado esperado:** reduz significativamente o tamanho efetivo do dataset de treino (~17%), mas cada epoch fica mais rápida e o gradiente tem melhor sinal.

**Onde:** [app/infrastructure/segmentation_dataset.py:115-158](app/infrastructure/segmentation_dataset.py).

---

### 2.4 Normalização ImageNet

**O que é:** transformação aplicada na imagem RGB antes de entrar no modelo, ajustando média e desvio-padrão para os valores usados no pré-treino do ImageNet.

**Como funciona (simples):** subtrai a média e divide pelo desvio-padrão, canal a canal:
- R: média=0.485, desvio=0.229
- G: média=0.456, desvio=0.224
- B: média=0.406, desvio=0.225

**Por que aplicado:** redes de visão computacional treinam melhor com entradas centradas em zero e variância similar entre canais. Usar as estatísticas do ImageNet é o padrão da literatura, mesmo quando não se faz transfer learning — facilita a inicialização e estabiliza o treinamento.

**Onde:** [app/infrastructure/segmentation_dataset.py:48-52](app/infrastructure/segmentation_dataset.py#L48-L52).

---

## 3. Augmentação de Dados

> **Princípio geral:** augmentação cria "novas" imagens de treino aplicando transformações aleatórias nas existentes. Aumenta a diversidade vista pelo modelo sem precisar coletar mais dados, reduzindo a tendência de decoração (overfitting).

### 3.1 Flips horizontal e vertical

**O que é:** espelhar a imagem horizontalmente e/ou verticalmente, aleatoriamente (50% de chance cada).

**Por que aplicado:** ortofotos aéreas não têm orientação privilegiada — um formigueiro visto de cima continua sendo um formigueiro independente de qual lado é "norte". Multiplica o dataset efetivo por 4 (combinações de flips H × V) sem custo computacional.

---

### 3.2 Rotação aleatória pequena (±15°)

**O que é:** rotacionar a imagem por um ângulo aleatório entre -15° e +15°.

**Por que aplicado:** simula pequenas variações de captura do drone/avião — ortofotas nem sempre são geometricamente perfeitas. Pequenos ângulos preservam o conteúdo central.

**Cuidado:** rotação cria pixels novos nas bordas (preenchimento). Esses pixels podem entrar como "fundo" em vez de "ignorar", inflando o sinal de fundo. Mitigado pela classe ignore-index e pelo skip de batches 100% ignore (ver §6.4).

---

### 3.3 Color Jitter (variação fotométrica)

**O que é:** alterações aleatórias de brilho, contraste e saturação na imagem (não na máscara).

**Configuração:** ±20% brilho, ±20% contraste, ±10% saturação.

**Por que aplicado:** ortofotos vêm de voos diferentes, em horas diferentes do dia, com sensores diferentes. Variar fotometria simula essa diversidade de captura — o modelo aprende que **a estrutura geométrica do formigueiro é o sinal**, não a cor exata.

---

### 3.4 Elastic Transform (deformação elástica)

**O que é:** distorção localizada da imagem como se fosse um tecido sendo puxado em direções aleatórias.

**Configuração final:** alpha=25 (intensidade da deformação), sigma=4 (suavidade).

**Como funciona (simples):** gera um campo de deslocamento aleatório suavizado, e cada pixel é movido um pouco segundo esse campo. O efeito visual é parecido com olhar a imagem através de água ondulada.

**Por que aplicado:** formigueiros não têm forma rígida — variam de redondos a alongados a irregulares. Elastic transform cria essas variações de forma de modo aprendível.

---

### 3.5 Copy-Paste Augmentation

**O que é:** "recortar" um formigueiro de uma imagem positiva e "colar" em uma imagem negativa, criando uma nova imagem positiva sintética.

#### 3.5.1 Implementação básica

**Como funciona (simples):**
1. Ao construir o dataset, lista todas as tiles que contêm formigueiros (~600).
2. Quando o modelo está prestes a ver uma tile sem formigueiro, com probabilidade `p`, pega aleatoriamente uma tile com formigueiro ("doadora").
3. Identifica o formigueiro na doadora usando análise de componentes conexos (`scipy.ndimage.label`) — cada região vermelha conexa é um candidato.
4. Recorta a região do formigueiro (com padding) e cola na imagem negativa em posição aleatória.
5. Atualiza a máscara para marcar a região colada como classe 1 (formigueiro).

**Por que aplicado:** apenas multiplicar/repetir os 600 doadores via oversampling ensinou modelos a decorar os mesmos exemplos. Copy-paste cria **combinações genuinamente novas** — formigueiros do dataset 1 colados em fundos do dataset 2.

#### 3.5.2 Melhorias implementadas

Três falhas qualitativas foram identificadas e corrigidas:

##### **P1 — Bordas suaves (Gaussian alpha blending)**

**Problema:** o paste usava transição brusca de 1 pixel entre doador e fundo. Visualmente aparecia como uma "silhueta recortada". O modelo aprendia esse padrão como atalho ("se há borda nítida, é formigueiro") em vez de aprender o padrão visual real.

**Solução:** suavizar a borda do paste com filtro gaussiano (σ=1.5px), criando uma transição de ~3-4 pixels. A imagem RGB sofre o blending, mas **a máscara de label permanece binária** — o sinal de supervisão fica nítido enquanto o visual fica natural.

```python
alpha = ndimage.gaussian_filter(mask_crop, sigma=1.5)
blended = alpha * doador + (1 - alpha) * fundo
```

**Por que funciona:** o modelo é forçado a aprender a fronteira **semântica** do formigueiro (texturas, cores, formas reais) em vez da fronteira **artificial** de colagem.

##### **P2 — Filtro de tamanho do doador**

**Problema:** sem filtro, o algoritmo aceitava qualquer componente conexo da máscara, incluindo:
- Ruído de 1-2 pixels (artefatos de anotação)
- Erros de máscara que se espalhavam por linhas de cultivo (>5.000 pixels)

O segundo caso é especialmente prejudicial — ensina ao modelo um "shape prior" errado.

**Solução:** filtrar componentes por área antes da escolha aleatória — aceitar apenas regiões com área entre 30 e 5.000 pixels.

```python
valid_components = [c for c in todos if 30 <= area(c) <= 5000]
escolhido = random.choice(valid_components)
```

**Justificativa dos limites:** 5.000 pixels é o mesmo limite usado no pós-processamento (`max_anthill_region_px`), garantindo coerência entre treino e inferência. 30 pixels rejeita ruído mas mantém formigueiros pequenos.

##### **P3 — Flip e rotação 90° do doador**

**Problema:** os ~600 doadores apareciam sempre na mesma orientação. Com múltiplas epochs, cada doador apareceria várias vezes idêntico → memorização.

**Solução:** antes de colar, aplicar aleatoriamente:
- Flip horizontal (50% de chance)
- Flip vertical (50% de chance)
- Rotação de 0°, 90°, 180° ou 270° (75% de chance de não ser 0°)

Isso multiplica os ~600 doadores por até 16 variantes (2 flips × 4 rotações) → ~9.600 variações efetivas.

```python
if random.random() < 0.5: rgb_crop = np.fliplr(rgb_crop)  # idem máscara
if random.random() < 0.5: rgb_crop = np.flipud(rgb_crop)
k = random.randint(0, 3)
rgb_crop = np.rot90(rgb_crop, k)
```

**Nota:** ao contrário da rotação 90° aplicada à imagem inteira (descartada por gerar orientações de cena irreais), a rotação aplicada apenas ao **formigueiro recortado** é segura — formigueiros individuais não têm orientação privilegiada (são aproximadamente simétricos por rotação).

---

## 4. Arquitetura do Modelo

### 4.1 U-Net

**O que é:** rede neural convolucional especializada em segmentação semântica, originalmente proposta para imagens médicas (Ronneberger et al., 2015).

**Como funciona (simples):** a U-Net tem formato de "U":
- **Lado esquerdo (encoder):** comprime a imagem progressivamente, extraindo características cada vez mais abstratas mas perdendo resolução espacial. 5 níveis: 64 → 128 → 256 → 512 → 1024 filtros.
- **Lado direito (decoder):** "descomprime" reconstruindo resolução, combinando informação espacial original com características aprendidas.
- **Skip connections (atalhos):** ligam diretamente cada nível do encoder ao nível correspondente do decoder. Isso permite que o decoder use detalhes finos da imagem original que seriam perdidos só com o caminho do encoder→decoder.

**Por que aplicado:** U-Net é o padrão-ouro para segmentação com poucas classes e dataset moderado. Suas skip connections são particularmente úteis para detectar objetos pequenos (como formigueiros) em imagens grandes.

**Onde:** [app/domain/unet.py](app/domain/unet.py).

---

### 4.2 Normalização (Batch → Group Normalization)

**O que é:** camada de normalização que padroniza as ativações após cada convolução, estabilizando o treinamento.

**Evolução no projeto:** inicialmente (Run 05) adotou-se **Batch Normalization**, que normaliza as ativações usando média e desvio calculados **dentro de cada batch**. Sua introdução, em conjunto com o ConvTranspose2d, foi decisiva para a evolução das métricas. Porém, devido à limitação de memória da GPU (6 GB), o treinamento usa **batch size = 2** — regime no qual as estatísticas por batch ficam ruidosas e pouco confiáveis. Por isso, o BatchNorm foi posteriormente **substituído por Group Normalization (8 grupos)**, que normaliza grupos de canais de forma **independente do tamanho do batch**. O modelo final (Run 14) usa GroupNorm.

**Por que normalizar (em qualquer das variantes):**
- **Estabiliza o treinamento:** sem normalização, gradientes podem explodir ou sumir em redes profundas
- **Permite learning rate maior:** com normalização, o modelo aceita taxas de aprendizado maiores sem explodir
- **Atua como regularizador leve**

**Por que GroupNorm em vez de BatchNorm neste projeto:** com batch=2, o BatchNorm estima média/desvio a partir de apenas 2 amostras por passo, tornando a normalização instável; o GroupNorm não depende do tamanho do batch e preserva a estabilidade do treino.

---

### 4.3 ConvTranspose2d vs Bilinear Upsampling

**O que é:** método de "aumentar a resolução" no decoder da U-Net.

**Como funciona (simples):**
- **Bilinear:** interpolação fixa, similar ao zoom de uma imagem em editor — calcula pixels intermediários por média ponderada dos vizinhos. **Não tem parâmetros aprendíveis.**
- **ConvTranspose2d:** "convolução transposta" — operação aprendível que aumenta a resolução. **Tem parâmetros que o gradiente ajusta.**

**Por que aplicado:** com ConvTranspose2d, o modelo aprende **como** reconstruir a resolução de modo otimizado para a tarefa, não apenas "esticar" a imagem. Em conjunto com a normalização nas convoluções (introduzida como BatchNorm na Run 05 e posteriormente migrada para GroupNorm), este é um salto qualitativo significativo.

---

## 5. Funções de Custo (Loss)

> A **loss** é a fórmula matemática que mede "quão errado o modelo está". O treinamento ajusta os pesos para minimizar a loss. **A escolha da loss é a decisão mais crítica em problemas desbalanceados.**

### 5.1 Cross-Entropy Loss + class weights

**O que é:** loss padrão de classificação. Penaliza pixels classificados errado.

**Como funciona (simples):** para cada pixel, calcula `-log(probabilidade da classe correta)`. Quanto menor a probabilidade que o modelo deu para a classe certa, maior a penalidade.

**Class weights:** multiplica a penalidade da classe formigueiro por um fator para compensar o desbalanceamento.

**Limitação:** mesmo com class weights, a Cross-Entropy é "preguiçosa" — convergir para "tudo fundo" minimiza a loss média rapidamente sem aprender a detectar formigueiros de fato.

---

### 5.2 Focal Loss

**O que é:** variante da Cross-Entropy que **reduz o peso de exemplos fáceis** (Lin et al., 2017).

**Como funciona (simples):** multiplica a penalidade por `(1 - p)^γ`, onde `p` é a confiança do modelo na classe correta e `γ` (gamma) controla o quanto exemplos fáceis são "amortecidos". Com γ=2.0:
- Pixel onde o modelo tem 99% de certeza correta: penalidade × 0.0001 (quase zero)
- Pixel onde o modelo tem 50% de certeza: penalidade × 0.25 (peso significativo)
- Pixel onde o modelo tem 10% de certeza: penalidade × 0.81 (peso quase total)

**Por que aplicado:** força o modelo a focar nos casos **difíceis** (solo ambíguo perto de formigueiros, formigueiros pequenos), em vez de gastar energia melhorando casos triviais (solo limpo).

**Configuração:** γ=2.0, valor padrão da literatura.

---

### 5.3 Tversky Loss

**O que é:** loss de sobreposição (similar à Dice Loss) com controle separado de penalização para falsos positivos e falsos negativos.

**Fórmula:**
```
TL = 1 - TP / (TP + α·FP + β·FN)
```
Onde TP=true positive, FP=false positive, FN=false negative (somados em pixels do batch).

**Como funciona (simples):** mede a sobreposição entre predição e ground truth. Os parâmetros α (alfa) e β (beta) determinam o que dói mais errar:
- α=β=0.5 → equivalente à Dice Loss padrão (FP e FN penalizados igualmente)
- α=0.3, β=0.7 → falsos negativos doem 2.3× mais que falsos positivos → empurra Recall (detecção)

**Por que aplicado:** missing um formigueiro real (FN) é geralmente mais grave que dar um falso alarme (FP), porque formigueiros precisam de tratamento (controle químico/biológico). Tversky permite codificar essa preferência diretamente na loss.

---

### 5.4 Combined Tversky+Focal Loss

**O que é:** combinação ponderada das duas losses.

**Fórmula:**
```
Total = w_tversky · Tversky + w_focal · Focal
```

**Por que combinada:** Tversky pura pode colapsar no início do treino quando o desbalanceamento é extremo — o termo `α·FP` domina o denominador, o gradiente empurra `prob(formigueiro) → 0` em todos os pixels, e a loss trava num platô. Solução: usar Focal como **âncora** (estabiliza o início) e Tversky como **direcionador** (otimiza Recall).

**Onde:** [app/service/training_service.py:121-176](app/service/training_service.py).

---

### 5.5 Lovász Hinge Loss

**O que é:** surrogate convexa por partes para o índice de Jaccard (IoU), proposta por Berman et al. (CVPR 2018).

**Como funciona (simples):**
- Outras losses (CE, Focal, Tversky) otimizam **classificação por pixel**, melhorando IoU como efeito colateral.
- Lovász otimiza **diretamente uma aproximação convexa do IoU**. Ela ordena os erros do batch e calcula o gradiente do índice de Jaccard como uma função das ordenações.
- Em termos práticos: se o modelo predisse uma máscara levemente maior ou menor que o real, a Lovász penaliza proporcionalmente ao impacto disso no IoU.

**Por que aplicado:** IoU é a métrica final que realmente mede desempenho, então otimizar diretamente para isso faz sentido.

**Loss combinada:**
```
0.5 · Tversky + 0.3 · Lovász + 0.2 · Focal
```

**Implementação:** auto-contida (sem dependências externas), em [app/service/training_service.py:121-181](app/service/training_service.py#L121-L181).

**Atenção sobre escala:** Lovász produz valores numericamente maiores que Tversky+Focal. **Não compare loss absoluta entre runs com e sem Lovász** — o que importa é a tendência decrescente.

---

## 6. Otimização do Treinamento

### 6.1 Otimizador Adam

**O que é:** algoritmo de otimização que ajusta os pesos do modelo baseado nos gradientes calculados pela loss.

**Como funciona (simples):** Adam mantém uma "memória" de gradientes passados (momento) e adapta a taxa de aprendizado para cada peso individualmente. Funciona bem em quase qualquer problema sem precisar de muito ajuste.

**Configuração:** learning rate inicial = 1e-3 (padrão da literatura).

**Onde:** [app/service/training_service.py](app/service/training_service.py).

---

### 6.2 Schedulers de Learning Rate

> Scheduler é a estratégia que **diminui o learning rate ao longo do treino**. Começar grande permite aprender rápido; reduzir depois permite refinar sem oscilar.

#### 6.2.1 ReduceLROnPlateau

**O que é:** reduz o LR quando a val_loss para de melhorar.

**Configuração:** `factor=0.5, patience=5` — a cada 5 epochs sem melhora, o LR é dividido por 2.

**Limitação:** ReduceLR depende de uma val_loss "limpa" para detectar plateaus. Quando há augmentações fortes ou samplers ponderados, a val_loss fica ruidosa, e o scheduler reduz o LR cedo demais, congelando o aprendizado.

#### 6.2.2 CosineAnnealingLR

**O que é:** scheduler com calendário **fixo** que segue uma curva cosseno de `lr_max → lr_min` ao longo de todas as epochs.

**Como funciona (simples):**
- Epoch 0: LR = 1e-3
- Epoch 50 (metade): LR = 5e-4
- Epoch 100 (final): LR = 1e-6
- A curva é suave (cosseno), não há "saltos"

**Por que aplicado:** imune a ruído na val_loss — sempre segue o mesmo calendário, garantindo decaimento suave. Particularmente útil quando há augmentações fortes que tornam a val_loss volátil.

**Configuração:** `T_max=50` epochs, `eta_min=1e-6`.

---

### 6.3 Gradient Clipping

**O que é:** limita o tamanho máximo dos gradientes durante o backprop.

**Como funciona (simples):** se a norma L2 do vetor de gradientes excede 1.0, o vetor inteiro é redimensionado para ter exatamente norma 1.0 (sem mudar de direção).

**Por que aplicado:** sem normalização nas convoluções, gradientes podem explodir e produzir Inf/NaN, corrompendo os pesos. Com a normalização (GroupNorm), o problema é menor mas o clipping continua como rede de segurança.

**Configuração:** `max_norm=1.0`.

**Onde:** [app/service/training_service.py:412](app/service/training_service.py#L412).

---

### 6.4 Skip de batches problemáticos (NaN/Inf/all-ignore)

**O que é:** pular batches que contenham:
- Valores numéricos inválidos (NaN, Inf) na entrada ou saída do modelo
- Todos os pixels marcados como ignore (label=255)

**Por que aplicado:**
- **NaN/Inf de inputs:** podem aparecer por SSL handshake quebrado ou arquivos corrompidos
- **NaN/Inf de outputs:** podem aparecer durante instabilidade numérica (mitigado por normalização (GroupNorm) + clipping, mas o skip continua como segurança)
- **All-ignore:** após rotação aleatória, pode ocorrer da imagem rotacionada cair inteiramente fora da área anotada → loss seria NaN (divisão por zero no denominador da Tversky)

Em vez de quebrar o treino, esses batches são logados como warning e pulados. O modelo continua aprendendo com o restante.

**Onde:** [app/service/training_service.py:351-377](app/service/training_service.py).

---

## 7. Pós-processamento das Predições

> Após o modelo prever a probabilidade de cada pixel ser formigueiro, três filtros refinam o resultado antes da avaliação.

### 7.1 Threshold de confiança

**O que é:** mínima probabilidade que um pixel precisa para ser classificado como formigueiro.

**Como funciona (simples):**
- O modelo gera, para cada pixel, uma probabilidade entre 0 e 1 de ser formigueiro.
- Apenas pixels com probabilidade ≥ threshold são marcados como formigueiro.
- Threshold=0.5 → equivalente a "argmax simples" (a classe mais provável vence)
- Threshold=0.7 → modelo precisa estar 70% certo (mais conservador, menos FP, mais FN)

**Configuração padrão:** 0.5.

---

### 7.2 Filtro por tamanho de região conexa

**O que é:** após o threshold, identifica grupos contíguos de pixels classificados como formigueiro (componentes conexos) e descarta os que estão fora da faixa de tamanho razoável.

**Configuração:**
- `min_anthill_region_px = 100` — descartar grupos menores que 100 pixels (ruído)
- `max_anthill_region_px = 5000` — descartar grupos maiores que 5.000 pixels (provavelmente erro detectando linhas de cultivo, sombras de máquinas, ou faixas de solo avermelhado)

**Por que aplicado:**
- Formigueiros típicos numa tile 256×256 têm ~3.000-4.000 pixels (diâmetro ~70px)
- Grupos abaixo de 100px são quase sempre ruído (pixels isolados que vazaram pelo threshold)
- Grupos acima de 5.000px são quase sempre falsos positivos massivos em padrões geométricos do solo

**Implementação:** [app/service/validation_service.py](app/service/validation_service.py) usando `scipy.ndimage.label` para detectar componentes conexos.

---

### 7.3 Análise de Componentes Conexos

**O que é:** método clássico de visão computacional que agrupa pixels vizinhos da mesma classe em "regiões".

**Como funciona (simples):** começando de cada pixel marcado como formigueiro, expande para vizinhos imediatos (cima, baixo, esquerda, direita) que também sejam formigueiro, até esgotar. Cada grupo expandido é uma "região" e recebe um ID único.

**Por que aplicado:** permite filtros baseados em **propriedades das regiões** (tamanho, formato, posição) em vez de apenas pixels individuais — padrão da indústria para segmentação aplicada.

---

## 8. Métricas de Avaliação

> Duas avaliações são feitas em paralelo: por imagem (detecção) e por pixel (segmentação).

### 8.1 Pixel Accuracy

**O que é:** fração de pixels classificados corretamente.

**Fórmula:** `pixels_corretos / total_de_pixels_não_ignorados`

**Por que é enganosa:** com desbalanceamento de 99% fundo / 1% formigueiro, basta prever "tudo fundo" para ter 99% de accuracy. **Por isso esta métrica nunca é usada sozinha.**

---

### 8.2 IoU (Intersection over Union, Jaccard Index)

**O que é:** métrica padrão de sobreposição entre predição e ground truth.

**Fórmula:** `interseção / união`

**Como funciona (simples):**
- Interseção = pixels que estão marcados como formigueiro **na predição E no ground truth**
- União = pixels marcados como formigueiro **na predição OU no ground truth**
- IoU=1.0 → predição perfeita
- IoU=0.5 → metade da predição está certa
- IoU=0.0 → não há sobreposição

**Por classe:** calculamos IoU separado para fundo (~99%) e formigueiro (~30-35%). O IoU de formigueiro é o número que realmente mede a qualidade do modelo.

---

### 8.3 Dice Coefficient

**O que é:** métrica similar ao IoU, mas mais sensível a regiões pequenas.

**Fórmula:** `2 × interseção / (predição + ground truth)`

**Relação com IoU:** matematicamente, `Dice = 2·IoU / (1 + IoU)`. Sempre Dice ≥ IoU. Em segmentação médica e científica, Dice é frequentemente reportado em conjunto com IoU.

---

### 8.4 Precision, Recall e F1 (nível de imagem)

**O que é:** métricas de detecção binária (a imagem tem formigueiro ou não), usadas além das métricas pixel-a-pixel.

**Definições:**
- **TP (True Positive):** modelo detectou e tinha formigueiro mesmo
- **FP (False Positive):** modelo detectou mas não tinha (falso alarme)
- **FN (False Negative):** modelo não detectou mas tinha (formigueiro perdido)
- **TN (True Negative):** modelo não detectou e não tinha

**Métricas:**
- **Precision = TP / (TP + FP)** → "dos que detectei, quantos eram reais?"
- **Recall = TP / (TP + FN)** → "dos que eram reais, quantos detectei?"
- **F1 = 2·P·R / (P+R)** → média harmônica (penaliza desequilíbrio entre P e R)

**Trade-off:** aumentar Recall geralmente diminui Precision e vice-versa. F1 captura o equilíbrio.

---

### 8.5 Métricas globais vs por imagem

Duas estratégias diferentes de agregação são reportadas:

#### Por imagem ([validation_service.py](app/service/validation_service.py))
- Calcula IoU/Dice **separadamente para cada imagem**
- Tira a média dessas métricas no final
- Resultado típico: mIoU ≈ 0.44

#### Globalmente ([run_evaluate.py](run_evaluate.py))
- Acumula pixels TP, FP, FN, TN **através de todas as imagens**
- Calcula uma única métrica final sobre o agregado
- Resultado típico: mIoU ≈ 0.65

**Por que diferente:** a média de razões ≠ razão das médias. Imagens sem nenhum formigueiro (e sem predição) têm IoU=1.0 trivialmente, "puxando" a métrica por imagem para cima. A versão global é mais rigorosa para datasets desbalanceados (poucas imagens positivas).

**Convenção neste projeto:** ambos os números são reportados, com `evaluate_detections` (global) sendo o número de referência para comparações entre runs.

---

## Referências de Implementação

| Componente          | Arquivo                                                                          |
| ------------------- | -------------------------------------------------------------------------------- |
| Configurações       | [app/core/config.py](app/core/config.py)                                         |
| Logger              | [app/core/logging_config.py](app/core/logging_config.py)                         |
| Dataset + augmentações | [app/infrastructure/segmentation_dataset.py](app/infrastructure/segmentation_dataset.py) |
| DataLoaders + augmentação pipeline | [app/service/data_service.py](app/service/data_service.py)         |
| U-Net (arquitetura) | [app/domain/unet.py](app/domain/unet.py)                                         |
| Treino + losses     | [app/service/training_service.py](app/service/training_service.py)               |
| Validação online    | [app/service/validation_service.py](app/service/validation_service.py)           |
| Avaliação global    | [run_evaluate.py](run_evaluate.py)                                               |

---

## Referências Bibliográficas

> As técnicas listadas neste documento são fundamentadas nos seguintes trabalhos.

- **U-Net:** Ronneberger, Fischer & Brox (2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- **Batch Normalization:** Ioffe & Szegedy (2015) — usada nas Runs 05–11
- **Group Normalization:** Wu & He (2018) — adotada a partir da migração de normalização (modelo final, Run 14)
- **Focal Loss:** Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- **Tversky Loss:** Salehi et al. (2017) — *Tversky loss function for image segmentation using 3D fully convolutional deep networks*
- **Lovász Hinge Loss:** Berman, Triki & Blaschko (2018) — *The Lovász-Softmax loss: A tractable surrogate for the optimization of the IoU measure*
- **Adam Optimizer:** Kingma & Ba (2014) — *Adam: A Method for Stochastic Optimization*
- **Cosine Annealing:** Loshchilov & Hutter (2017) — *SGDR: Stochastic Gradient Descent with Warm Restarts*
- **Copy-Paste Augmentation:** Ghiasi et al. (2021) — *Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation*
- **Elastic Transform:** Simard, Steinkraus & Platt (2003) — *Best practices for convolutional neural networks applied to visual document analysis*
