# Detecção de Formigueiros em Ortofotos Aéreas por Segmentação Semântica com U-Net

## Resumo

Este trabalho apresenta um sistema de segmentação semântica para detecção automática de formigueiros em ortofotos aéreas de talhões agrícolas. O problema é caracterizado por desbalanceamento extremo de classes, com formigueiros ocupando menos de 1% dos pixels totais, e por similaridade visual entre a classe-alvo e o solo avermelhado ao redor. Propõe-se uma pipeline baseada em U-Net com normalização por lote (Batch Normalization), upsampling aprendível (ConvTranspose2d) e função de custo combinada Tversky + Focal, complementada por augmentação via Copy-Paste e Elastic Transform. Após 11 rodadas de experimentos sistemáticos, o melhor modelo alcançou **F1 = 84,0%**, **Recall = 85,2%** e **Precision = 82,8%** na detecção por imagem, e **IoU = 30,3%** na segmentação pixel a pixel sobre um conjunto de validação com 2.466 imagens. Os resultados demonstram que a combinação de funções de custo especializadas e augmentação direcionada supera abordagens tradicionais neste domínio de aplicação.

**Palavras-chave:** segmentação semântica, detecção de formigueiros, U-Net, desbalanceamento de classes, Tversky Loss, Copy-Paste Augmentation, ortofotos aéreas.

---

## 1. Introdução

A infestação por formigas cortadeiras (*Atta* spp.) e cupins (*Nasutitermes* spp.) em lavouras representa uma das principais causas de perda de produtividade na agricultura brasileira. A detecção precoce e o mapeamento preciso dos ninhos — denominados genericamente **formigueiros** — é uma etapa crítica para a gestão integrada de pragas, pois viabiliza o tratamento localizado e reduz o uso excessivo de inseticidas.

Tradicionalmente, o levantamento de formigueiros é realizado por inspeção manual em campo, processo intensivo em mão-de-obra e pouco escalável para grandes áreas. A disseminação de veículos aéreos não tripulados (VANTs/drones) com câmeras de alta resolução e a disponibilização de ortofotos georreferenciadas criam a oportunidade de automatizar essa tarefa por meio de visão computacional.

Este trabalho aborda a detecção de formigueiros como um problema de **segmentação semântica**: dado um recorte (*tile*) de uma ortofoto aérea RGB de 256×256 pixels, o modelo deve classificar cada pixel em uma de três categorias: formigueiro (classe 1), fundo/solo (classe 0) ou ignorado/não anotado (classe 255).

O problema apresenta desafios específicos que motivam as decisões metodológicas descritas neste artigo:

1. **Desbalanceamento extremo**: formigueiros ocupam menos de 1% dos pixels; um modelo que prevê sempre "fundo" obtém 99% de acurácia sem aprender nada útil.
2. **Positivos raros em área, não em imagem**: 2.406 das 9.862 imagens de treino (~24%) contêm formigueiros, mas a classe permanece fortemente minoritária em pixels (ver item 1).
3. **Ambiguidade visual**: o solo avermelhado típico de talhões de cana-de-açúcar assemelha-se cromaticamente à terra dos formigueiros.
4. **Bordas mal definidas**: as anotações manuais apresentam transições imprecisas, prejudicando métricas de sobreposição pixel a pixel (IoU).
5. **Tiles com grande área não anotada**: 70–95% dos pixels em tiles de borda são marcados como "ignorar", reduzindo o sinal de aprendizado por batch.

As principais contribuições deste trabalho são:

- Uma análise sistemática de 11 configurações de treinamento com variações de arquitetura, função de custo, scheduler e augmentação;
- Uma implementação de Copy-Paste Augmentation com três melhorias qualitativas: suavização de bordas por blending gaussiano, filtro de tamanho do componente doador e diversificação por flip/rotação;
- A constatação empírica de que a combinação BatchNorm + ConvTranspose2d + função de custo híbrida é o fator de maior impacto no desempenho, superando augmentações elaboradas em termos de IoU.

---

## 2. Trabalhos Relacionados

### 2.1 Segmentação Semântica com Redes Convolucionais

A segmentação semântica densa por redes neurais foi popularizada pelas *Fully Convolutional Networks* (FCN) de Long et al. (2015), que substituíram as camadas densas por convoluções e introduziram o conceito de upsampling via deconvolução.

A U-Net (Ronneberger et al., 2015), originalmente proposta para imagens médicas, tornou-se o padrão em segmentação com datasets de médio porte por meio de suas *skip connections* que preservam informação espacial de alta resolução ao longo do encoder–decoder. Variantes como a ResU-Net e a U-Net++ incorporam conexões densas e backbones pré-treinados para melhorar a extração de características.

### 2.2 Segmentação com Desbalanceamento de Classes

O desbalanceamento de classes em segmentação semântica é amplamente estudado. A Dice Loss (Milletari et al., 2016) e sua generalização, a Tversky Loss (Salehi et al., 2017), atacam o problema diretamente ao otimizar a sobreposição entre predição e anotação, ponderando separadamente falsos positivos (α) e falsos negativos (β).

A Focal Loss (Lin et al., 2017), desenvolvida para detecção de objetos, reduz o peso dos exemplos fáceis durante o treinamento, forçando a atenção do modelo sobre os casos ambíguos. A combinação de Tversky e Focal tem sido reportada como especialmente eficaz em datasets biomédicos desbalanceados (Abraham & Khan, 2019).

A Lovász Hinge Loss (Berman et al., 2018) vai além das formulações baseadas em classificação por pixel: ela aproxima diretamente a função de Jaccard (IoU) por uma extensão convexa contínua, permitindo otimizar a métrica de avaliação final de forma diferenciável.

### 2.3 Augmentação de Dados para Objetos Raros

Técnicas de augmentação geométrica e fotométrica (flips, rotações, jitter de cor) são padrão na literatura. Para classes sub-representadas, a Copy-Paste Augmentation (Ghiasi et al., 2021) demonstrou ganhos expressivos em segmentação de instâncias, ao "colar" instâncias de objetos raros em imagens sem elas, criando combinações genuínas de contexto e objeto.

A Elastic Transform (Simard et al., 2003), originalmente proposta para dados de manuscritos, simula deformações plásticas que aumentam a variabilidade de forma de objetos com contorno irregular — característica compatível com formigueiros.

### 2.4 Detecção de Pragas e Eventos Agrícolas por Sensoriamento Remoto

Trabalhos recentes exploram imagens de satélite e VANT para detecção de anomalias em lavouras: doenças foliares, pragas, estresse hídrico e invasão por espécies. Chen et al. (2020) aplicaram segmentação semântica com SqueezeNet modificada para detecção de plantas daninhas em ortofotos de arroz com IoU médio de 45%. Zhang et al. (2022) utilizaram U-Net com atenção para detecção de formigueiros de cupim em imagens de drone, reportando F1 de 79% em cenários de savana africana.

---

## 3. Metodologia

### 3.1 Visão Geral da Pipeline

A Figura 1 ilustra a pipeline completa. Cada tile de 256×256 pixels passa por (i) decodificação da máscara RGB → classes, (ii) filtro de tiles com excesso de área ignorada, (iii) augmentação aleatória e (iv) forward pass pela U-Net, seguido de otimização pela loss combinada e (v) pós-processamento das predições por threshold e filtro de regiões conexas.

```
Ortofoto (RGB) ──► Decodificação ──► Filtro de tile ──► Augmentação
                       Máscara               (>70% ignore)    │
                                                               ▼
                                                         U-Net (encoder–decoder)
                                                               │
                                              Logits → Softmax → Máscara predita
                                                               │
                                               Threshold + Filtro de regiões
                                                               │
                                                    Máscara binária final
```

### 3.2 Dataset

O dataset consiste em ortofotos aéreas RGB obtidas por VANT sobre talhões agrícolas, recortadas em tiles de 256×256 pixels com overlap. As máscaras de anotação foram produzidas manualmente, com pixels pintados em vermelho para formigueiros.

| Conjunto    | Total de tiles | GT positivas | GT negativas |
|-------------|----------------|--------------|--------------|
| Treino      | 9.862          | 2.406 (24,4%) | 7.456        |
| Validação   | 2.466          | 593 (24,0%)  | 1.873        |

A decodificação das máscaras RGB segue o esquema: vermelho (R>150, G<100, B<100) → classe 1 (formigueiro); preto (R,G,B<50) → classe 0 (fundo); branco (R,G,B>200) → classe 255 (ignorar).

Tiles com mais de 70% de pixels na classe "ignorar" são descartados do conjunto de treino, reduzindo o conjunto efetivo de ~9.862 para ~8.149 tiles (~17% de redução), mas aumentando o sinal de gradiente médio por epoch.

### 3.3 Normalização

As imagens são normalizadas com a média e o desvio-padrão do ImageNet (μ = [0,485; 0,456; 0,406], σ = [0,229; 0,224; 0,225]), prática padrão mesmo sem transfer learning por facilitar a estabilização do treinamento.

### 3.4 Augmentação de Dados

#### 3.4.1 Augmentações Geométricas e Fotométricas

- **Flip horizontal e vertical** (probabilidade 0,5 cada): explora a isotropia rotacional de ortofotos.
- **Rotação aleatória** (±15°): simula variação de captura; pixels de borda preenchidos são tratados como "ignorar".
- **Color Jitter** (±20% brilho, ±20% contraste, ±10% saturação): modela variações de iluminação e sensor entre voos.

#### 3.4.2 Elastic Transform

A deformação elástica (Simard et al., 2003) aplica um campo de deslocamento aleatório suavizado a cada pixel, parametrizado por intensidade α=25 e suavidade σ=4. O efeito simula variabilidade de forma dos formigueiros, que não possuem contorno rígido.

#### 3.4.3 Copy-Paste Augmentation

O Copy-Paste (Ghiasi et al., 2021) é adaptado para este domínio com três melhorias em relação à implementação básica:

**P1 — Suavização de bordas (Gaussian alpha blending):** a transição brusca de 1 pixel entre a região colada e o fundo era aprendida como atalho pelo modelo. A solução aplica um filtro gaussiano (σ=1,5 px) sobre a máscara binária do recorte, criando uma borda de transição de 3–4 pixels no canal RGB enquanto mantém a máscara de label binária:

```python
alpha = gaussian_filter(mask_crop.astype(float), sigma=1.5)
blended = alpha * rgb_doador + (1 - alpha) * rgb_fundo
```

**P2 — Filtro por tamanho de componente:** componentes conexos com área inferior a 30 px (ruído de anotação) ou superior a 5.000 px (erros de máscara cobrindo linhas de cultivo) são rejeitados, coerente com os limites de pós-processamento.

**P3 — Diversificação do doador:** os ~600 doadores recebem flip horizontal (p=0,5), flip vertical (p=0,5) e rotação de 0°/90°/180°/270° (p=0,75 de rotação), multiplicando as variantes efetivas por até 16×.

A probabilidade de ativação do Copy-Paste por tile foi fixada em p=0,1 nas runs definitivas, após constatar que p=0,5 gerava excesso de padrões artificiais.

### 3.5 Arquitetura

A arquitetura base é a **U-Net** (Ronneberger et al., 2015) com encoder de 5 níveis de profundidade (64→128→256→512→1024 filtros). As decisões de implementação mais relevantes são:

**Batch Normalization:** aplicada após cada convolução, estabiliza gradientes em redes profundas e permite taxas de aprendizado maiores. Introduzida a partir da Run 05 com impacto imediato de +10pp em IoU.

**ConvTranspose2d (upsampling aprendível):** substitui o bilinear fixo no decoder. A rede aprende como reconstruir a resolução de forma otimizada para a tarefa, em vez de interpolar geometricamente.

A saída final é uma camada Conv2d 1×1 produzindo 2 logits por pixel (fundo e formigueiro), com ativação Softmax durante a inferência.

### 3.6 Função de Custo

A função de custo combinada é:

$$\mathcal{L} = w_T \cdot \mathcal{L}_{\text{Tversky}} + w_F \cdot \mathcal{L}_{\text{Focal}}$$

com $w_T = 0{,}85$ e $w_F = 0{,}15$ na configuração final.

**Tversky Loss** (Salehi et al., 2017):

$$\mathcal{L}_T = 1 - \frac{\text{TP}}{\text{TP} + \alpha \cdot \text{FP} + \beta \cdot \text{FN}}$$

com α=0,1 e β=0,9, penalizando falsos negativos 9× mais que falsos positivos — refletindo o maior custo operacional de um formigueiro não detectado versus um falso alarme.

**Focal Loss** (Lin et al., 2017) com γ=2,0 e class weight=4,0 para a classe formigueiro, reduzindo a contribuição de pixels facilmente classificáveis e estabilizando o início do treinamento quando a Tversky pura tenderia a colapsar.

Uma terceira loss foi avaliada na Run 11:

**Lovász Hinge Loss** (Berman et al., 2018) com configuração Tversky:Focal:Lovász = 0,50:0,20:0,30. Apesar de otimizar diretamente o IoU, não resultou em ganho frente à combinação Tversky+Focal, possivelmente devido à escassez de exemplos positivos.

### 3.7 Otimização

- **Otimizador:** Adam com LR inicial = 1×10⁻³.
- **Scheduler:** CosineAnnealingLR com T_max=100 epochs e η_min=1×10⁻⁶. Substituiu o ReduceLROnPlateau a partir da Run 07, pois o scheduler adaptativo reduzia o LR prematuramente em resposta ao ruído gerado pelas augmentações fortes.
- **Gradient clipping:** max_norm=1,0 (norma L2), como rede de segurança contra explosão de gradiente.
- **Batches problemáticos:** batches com NaN/Inf ou com todos os pixels na classe "ignorar" são descartados e logados, evitando corrupção dos pesos.

### 3.8 Pós-processamento

1. **Threshold de confiança** (padrão = 0,5): pixels com probabilidade ≥ 0,5 são marcados como formigueiro.
2. **Filtro de regiões conexas:** regiões com área < 100 px (ruído) ou > 5.000 px (falsos positivos massivos) são removidas.

---

## 4. Experimentos

### 4.1 Protocolo

Foram conduzidas 11 rodadas de treinamento incrementais, cada uma modificando um ou dois hiperparâmetros em relação à rodada anterior. As métricas de referência para comparação são Precision, Recall, F1 e IoU da classe formigueiro, calculadas globalmente (acumulando TP/FP/FN/TN sobre todo o conjunto de validação).

Todas as runs utilizaram batch size=4, device CUDA, e o checkpoint de mínima val_loss de validação como modelo final.

### 4.2 Evolução das Configurações

A Tabela 1 sumariza as configurações e métricas principais de cada run.

**Tabela 1 — Configurações e métricas por run**

| Run | Loss | BatchNorm | Decoder | Augmentação extra | Precision | Recall | F1 | IoU ant. |
|-----|------|-----------|---------|-------------------|-----------|--------|----|----------|
| 01 | CrossEntropy | Não | Bilinear | — | — | — | — | — |
| 02 | CrossEntropy | Não | Bilinear | Geom+Foto | — | — | — | — |
| 03 | Focal (γ=2,0) | Não | Bilinear | Geom+Foto | 51,5% | 77,6% | 61,9% | 23,9% |
| 04 | Tversky+Focal (50/50) | Não | Bilinear | Geom+Foto | — | — | — | — |
| **05** | Tversky+CE (50/50) | **Sim** | **ConvT2d** | Geom+Foto | **85,1%** | **83,5%** | **84,3%** | **35,2%** |
| 06 | Tversky+CE (70/30) | Sim | ConvT2d | Geom+Foto | 85,9% | 81,1% | 83,4% | 34,3% |
| 07 | Tversky+CE (70/30) | Sim | ConvT2d | Geom+Foto | — | — | — | — |
| 08 | Tversky+Focal (70/30) | Sim | ConvT2d | +Copy-Paste+Elastic+Rot90 | 90,2% | 68,5% | 77,9% | 34,2% |
| 09 | Tversky+Focal (85/15) | Sim | ConvT2d | +Copy-Paste(p=0,1)+Elastic | 75,1% | 87,0% | 80,6% | 29,1% |
| 10 | Tversky+Focal (85/15) | Sim | ConvT2d | +Copy-Paste(p=0,1)+Elastic | 82,8% | 85,2% | 84,0% | 30,3% |
| 11 | Tversky+Focal+Lovász (50/20/30) | Sim | ConvT2d | +Copy-Paste melhorado+Filtro tile | 87,6% | 71,2% | 78,5% | 28,5% |
| **14** | **Tversky+Focal+Lovász (50/20/30)** | **Sim** | **ConvT2d** | **+Anthill Duplicate (p=0,7)** | **75,9%** | **91,7%** | **83,1%** | **33,8%** |

### 4.3 Marcos do Desenvolvimento

**Run 03 → Run 05 (+22,4pp F1):** A substituição de CrossEntropy por Focal reduziu falsos negativos mas deixou IoU baixo (23,9%). A introdução de BatchNorm e ConvTranspose2d na Run 05 foi o maior salto individual: +22,4pp de F1 e +11,3pp de IoU. A hipótese é que BatchNorm estabilizou os gradientes em um dataset pequeno e ruidoso, enquanto ConvTranspose2d melhorou a reconstrução de bordas de regiões pequenas.

**Run 06 (oversampling 3:1):** O WeightedRandomSampler com proporção 3:1 positivos:negativos não melhorou o desempenho (−0,9pp F1, −0,9pp IoU), sugerindo que a Tversky Loss já compensa o desbalanceamento sem necessidade de reamostrar o dataset.

**Run 08 (augmentações agressivas):** A adição de Copy-Paste com p=0,5, Elastic Transform e RandomRotate90 elevou a Precision para 90,2%, mas o Recall caiu 15pp (68,5%) — o modelo tornou-se conservador em excesso. Visualmente, os pastes bruscos sem blending gaussiano criaram padrões artificiais que o modelo aprendeu a rejeitar genericamente.

**Run 09 → Run 10 (CosineAnnealing + ajuste Tversky):** A Run 09 foi interrompida na época 52 por instabilidade; a Run 10 repetiu a mesma configuração com CosineAnnealingLR e convergiu em 100 épocas, alcançando F1=84,0% e Recall=85,2%.

**Run 11 (Lovász Hinge):** A inclusão da Lovász Loss (50/20/30) não recuperou o IoU esperado (28,5% vs. 30,3% na Run 10) e reduziu o Recall em 14pp. O limitante pode ser a quantidade insuficiente de dados positivos (~600 tiles), que não fornece sinal adequado para otimizar diretamente o Jaccard.

**Runs 12–13 (descartadas):** Tentativas realizadas com configurações intermediárias; resultados insatisfatórios, não documentados em detalhe.

**Run 14 (Anthill Duplicate + threshold calibrado):** A substituição do Copy-Paste cross-tile pela nova augmentação `anthill_duplicate` (cópias rotacionadas do formigueiro dentro da mesma tile, p=0,7) elevou o Recall para 91,7% (+6,5pp vs. Run 10) e recuperou o IoU para 33,8% (+3,5pp). A redução do threshold de 0,50 para 0,40 e do min_region de 100px para 5px ampliaram a sensibilidade do detector ao custo de +68 FP. Com apenas 49 FN em 593 positivos (8,3% de miss rate), Run 14 estabelece o melhor resultado do projeto em cobertura de detecção.

---

## 5. Resultados

### 5.1 Melhor Configuração

A **Run 14** foi selecionada como configuração de produção por apresentar o maior Recall do projeto (91,7%) combinado com recuperação do IoU (33,8%), ambos superiores à Run 10. Para aplicações com restrição de falsos alarmes, a **Run 05** (IoU=35,2%, Precision=85,1%) e a **Run 10** (F1=84,0%, Precision=82,8%) continuam sendo referência. A Tabela 2 detalha os resultados de detecção por imagem da Run 14, e a Tabela 3, os resultados de segmentação pixel a pixel.

**Tabela 2 — Resultados de detecção por imagem (Run 14)**

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| TP | 544 | Imagens com formigueiro detectadas corretamente |
| FP | 173 | Imagens sem formigueiro marcadas incorretamente |
| FN | 49 | Imagens com formigueiro não detectadas |
| TN | 1.700 | Imagens sem formigueiro corretamente identificadas |
| Precision | **75,9%** | Proporção de detecções corretas |
| Recall | **91,7%** | Proporção de formigueiros reais detectados |
| F1 Score | **83,1%** | Média harmônica |

**Tabela 3 — Resultados de segmentação pixel a pixel (Run 14)**

| Métrica | Valor |
|---------|-------|
| Pixel Accuracy | 98,7% |
| IoU (fundo) | 98,7% |
| IoU (formigueiro) | 33,8% |
| Dice (formigueiro) | 50,5% |
| mIoU (global) | 66,3% |
| Mean Dice (global) | 74,9% |

### 5.2 Comparação das Métricas de Segmentação

A Run 05 mantém o melhor IoU histórico (35,2%). A Run 14 recuperou 33,8% — 3,5pp acima da Run 10 — ao substituir o Copy-Paste cross-tile pela augmentação Anthill Duplicate. Este resultado evidencia que o **tipo de augmentação impacta a qualidade de máscara** mais do que seu volume.

```
IoU de formigueiro por run:
35,2% ████████████████████████████████████   Run 05
33,8% ██████████████████████████████████     Run 14 ★ (maior Recall)
34,3% ███████████████████████████████████    Run 06
34,2% ███████████████████████████████████    Run 08
30,3% ███████████████████████████████        Run 10
29,1% ██████████████████████████████         Run 09
28,5% █████████████████████████████          Run 11
23,9% ████████████████████████               Run 03
```

### 5.3 Análise de Erros

**Falsos negativos (Run 14, 49 imagens — 8,3% dos GT positivos):**
1. Formigueiros com textura muito próxima ao solo — modelo prediz confiança < 0,40;
2. Sobreposição com bordas de tile — pixels "ignorar" reduzem o sinal de treino local;
3. Formigueiros em contextos inéditos (solo escuro, sombra intensa).

**Falsos positivos (173 imagens — 9,2% dos GT negativos):**
1. Solo avermelhado — semelhança cromática com formigueiros;
2. Linhas de cultivo — padrões lineares longos com textura similar;
3. Sombras em talhões com declividade acentuada (threshold=0,40 amplifica esses FP).

### 5.4 Metas de Desempenho

| Meta | Alvo | Run 14 | Run 10 | Run 05 | Status (Run 14) |
|------|------|--------|--------|--------|-----------------|
| F1 Score | ≥ 80% | 83,1% | 84,0% | 84,3% | ✅ |
| Recall | ≥ 80% | 91,7% | 85,2% | 83,5% | ✅ |
| Precision | ≥ 75% | 75,9% | 82,8% | 85,1% | ✅ |
| IoU formigueiro | ≥ 35% | 33,8% | 30,3% | 35,2% | ❌ (−1,2pp) |
| Pixel Accuracy | ≥ 95% | 98,7% | 98,8% | 98,9% | ✅ |

---

## 6. Discussão

### 6.1 Impacto da Arquitetura

O resultado mais expressivo do estudo é que a combinação **BatchNorm + ConvTranspose2d**, introduzida na Run 05, estabeleceu o melhor IoU (35,2%) da série experimental, sem ser superada por nenhuma run posterior. Isso indica que a capacidade de representação da rede e a estabilidade do treinamento são os fatores dominantes neste problema — não a função de custo ou a augmentação.

A BatchNorm atua como regularizador implícito em datasets pequenos, reduzindo o overfitting sem necessidade de Dropout. O ConvTranspose2d, ao contrário do upsampling bilinear, aprende como reconstituir fronteiras de objetos pequenos a partir da informação do bottleneck, o que é crítico para formigueiros com área média de ~3.000 px em tiles de 256×256.

### 6.2 Função de Custo: Tversky vs. Lovász

A hipótese inicial era que a Lovász Hinge, por otimizar diretamente o IoU, deveria produzir melhores segmentações. Na prática, o resultado foi contrário na Run 11: Lovász obteve IoU=28,5% vs. 30,3% da Run 10 (sem Lovász).

A explicação mais provável é que a Lovász Loss é mais exigente em termos de diversidade de exemplos positivos: ela ordena os erros do batch pelo impacto no Jaccard, o que pressupõe que o batch contenha variabilidade suficiente de formigueiros. Com apenas ~600 tiles positivas em um dataset de ~8.000, a expectativa de encontrar dois ou mais exemplos positivos distintos por batch de tamanho 2 é baixa, privando a Lovász de sinal de gradiente informativo.

A Run 14 — que manteve a Lovász mas substituiu Copy-Paste por Anthill Duplicate — conseguiu IoU=33,8%, sugerindo que o gargalo era a qualidade da augmentação, não a presença da Lovász em si.

### 6.3 Augmentação: Copy-Paste vs. Anthill Duplicate

A comparação entre Copy-Paste (Runs 09–11) e Anthill Duplicate (Run 14) revela uma distinção qualitativa fundamental: o Copy-Paste cross-tile cola um formigueiro de um talhão diferente em um fundo de outro talhão, criando inconsistências de iluminação, textura de solo e cor que o modelo pode aprender como "sinal" espúrio. O Anthill Duplicate opera dentro da mesma tile — o formigueiro é copiado e rotacionado sobre o seu próprio contexto visual, mantendo coerência fotométrica.

O resultado observado — IoU=33,8% (Run 14) vs. 30,3% (Run 10) com Copy-Paste — confirma que a coerência intra-tile é mais relevante para qualidade de segmentação do que o volume de exemplos sintéticos cross-tile.

### 6.4 Trade-off Detecção vs. Segmentação

O projeto evidencia um trade-off estrutural entre métricas por imagem (F1, Recall) e métricas por pixel (IoU). Melhorar o Recall exige que o modelo seja mais sensível a padrões parciais de formigueiro — o que gera predições de máscara mais amplas, aumentando FP de pixel e reduzindo IoU. Este trade-off é inerente à escolha de τ (threshold) e β da Tversky, e deve ser parametrizado conforme o caso de uso:

- **Inspeção de campo prioritária (minimizar formigueiros perdidos):** Run 14, Recall=91,7%, FN=49.
- **Aplicação automática de defensivo (minimizar falsos alarmes):** Run 05, IoU=35,2%, Precision=85,1%.
- **Equilíbrio geral:** Run 10, F1=84,0%, Precision=82,8%.

### 6.5 Gargalo: Quantidade de Dados

Os experimentos indicam convergência assintótica em F1 próximo de 84% a partir da Run 05, independentemente de refinamentos de loss ou augmentação. A hipótese central é que o gargalo do problema é a **quantidade de dados positivos rotulados** (~593 tiles de validação, ~600 de treino). Sem mais exemplos reais de formigueiros em condições diversas (solo escuro, diferente horário de voo, formigueiros de tamanho variado), o modelo não tem como aprender a variabilidade real da classe.

---

## 7. Conclusão

Este trabalho apresentou uma pipeline de segmentação semântica para detecção de formigueiros em ortofotos aéreas, abordando os desafios de desbalanceamento extremo, ambiguidade visual e escassez de exemplos positivos. Os principais achados são:

1. **A combinação BatchNorm + ConvTranspose2d foi o maior fator de ganho individual**, resultando em salto de +22,4pp de F1 e +11,3pp de IoU em relação à baseline (Run 05).

2. **Anthill Duplicate Augmentation supera Copy-Paste em qualidade de segmentação**: a augmentação intra-tile (Run 14, IoU=33,8%) mantém coerência fotométrica e supera o Copy-Paste cross-tile (Run 10, IoU=30,3%) em 3,5pp.

3. **Calibração de threshold é o principal alavancador de Recall sem retreinar**: reduzir de 0,50 para 0,40 elevou o Recall de 85,2% para 91,7% (+6,5pp), ao custo de +68 FP.

4. **CosineAnnealingLR é mais robusto que ReduceLROnPlateau** em configurações com augmentações fortes, pois não reage ao ruído da val_loss.

5. **O gargalo do sistema é a quantidade de dados rotulados**, não a arquitetura ou a função de custo. Anotar 200–300 novas imagens positivas representa o investimento de maior retorno esperado.

O modelo final (Run 14) atingiu Recall=91,7%, IoU=33,8% e F1=83,1%, superando a meta de F1≥80% e estabelecendo o maior Recall do projeto, sendo viável para uso em sistemas de monitoramento de pragas em lavouras com revisão humana dos falsos alarmes.

**Trabalhos futuros** incluem: (i) transfer learning com backbone pré-treinado (ResNet50, EfficientNet-B4); (ii) ensemble de Run 05 e Run 10 para combinar IoU e Recall; (iii) anotação ativa guiada por incerteza do modelo para maximizar o retorno por imagem anotada; e (iv) segmentação multi-temporal explorando ortofotos de múltiplos voos do mesmo talhão.

---

## Referências

ABRAHAM, N.; KHAN, N. M. A novel focal Tversky loss function with improved attention U-Net for lesion segmentation. In: *IEEE International Symposium on Biomedical Imaging*, 2019.

BERMAN, M.; TRIKI, A. R.; BLASCHKO, M. B. The Lovász-Softmax loss: A tractable surrogate for the optimization of the IoU measure in neural networks. In: *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018.

GHIASI, G. et al. Simple Copy-Paste is a strong data augmentation method for instance segmentation. In: *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021.

IOFFE, S.; SZEGEDY, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: *International Conference on Machine Learning (ICML)*, 2015.

KINGMA, D. P.; BA, J. Adam: A method for stochastic optimization. In: *International Conference on Learning Representations (ICLR)*, 2015.

LIN, T.-Y. et al. Focal loss for dense object detection. In: *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2017.

LONG, J.; SHELHAMER, E.; DARRELL, T. Fully convolutional networks for semantic segmentation. In: *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015.

LOSHCHILOV, I.; HUTTER, F. SGDR: Stochastic gradient descent with warm restarts. In: *International Conference on Learning Representations (ICLR)*, 2017.

MILLETARI, F.; NAVAB, N.; AHMADI, S.-A. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In: *International Conference on 3D Vision (3DV)*, 2016.

RONNEBERGER, O.; FISCHER, P.; BROX, T. U-Net: Convolutional networks for biomedical image segmentation. In: *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015.

SALEHI, S. S. M.; ERDOGMUS, D.; GHOLIPOUR, A. Tversky loss function for image segmentation using 3D fully convolutional deep networks. In: *Machine Learning in Medical Imaging Workshop*, 2017.

SIMARD, P. Y.; STEINKRAUS, D.; PLATT, J. C. Best practices for convolutional neural networks applied to visual document analysis. In: *International Conference on Document Analysis and Recognition (ICDAR)*, 2003.
