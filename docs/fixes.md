# Correções no Pipeline de Dados

Este documento descreve os principais problemas identificados no pipeline de dados para segmentação semântica de formigueiros, bem como as soluções implementadas e seus impactos no desempenho do modelo.

## 1. Transformações desalinhadas

### Problema

As transformações geométricas (rotação, flip, entre outras) eram aplicadas de forma independente na imagem e na máscara.

### Impacto

- Desalinhamento espacial entre imagem e rótulo
- Violação da correspondência pixel a pixel exigida pelo aprendizado supervisionado
- Degradação significativa do desempenho (IoU não evoluía)

### Solução

Aplicação sincronizada das transformações:

```python
image, mask = transforms(image, mask)
```

## 2. Ausência de normalização

### Problema

As imagens eram utilizadas sem padronização dos valores de entrada.

### Impacto

- Distribuição de dados inconsistente
- Instabilidade no processo de otimização
- Convergência mais lenta

### Solução

Aplicação de normalização baseada em estatísticas do ImageNet, incluindo conversão para float32 e escala para o intervalo [0, 1]:

```python
image = TF.normalize(image, mean, std)
```

## 3. Valores inválidos nas máscaras

### Problema

Após as transformações, as máscaras apresentavam valores inconsistentes (diferentes de 0, 1 ou 255).

### Impacto

- Erros no cálculo da função de perda
- Introdução de ruído no gradiente
- Comprometimento do aprendizado supervisionado

### Solução

Padronização dos valores das máscaras:

```python
label = np.where(label > 1, 255, label)
```

## 4. Desbalanceamento de classes

### Problema

Predominância de pixels de fundo no dataset (~90%), com baixa representatividade da classe de interesse (formigueiros).

### Impacto

- Viés do modelo para a classe majoritária
- Redução da sensibilidade à classe minoritária
- IoU limitado (~0.34) e baixo recall

### Solução

Aplicação da técnica de Patch Training, com extração de recortes de tamanho fixo (512×512 pixels), priorizando regiões com maior presença de formigueiros.

Impacto da abordagem:

- Aumento da densidade de pixels positivos por amostra
- Redução do desbalanceamento local
- Melhoria na capacidade de aprendizado da classe minoritária

## 5. Patches sem informação relevante

### Problema

Alguns recortes extraídos não continham pixels da classe de interesse.

### Impacto

- Treinamento ineficiente
- Persistência do viés para a classe de fundo

### Solução

- Realização de múltiplas tentativas de extração por imagem
- Seleção do patch com maior quantidade de pixels de formigueiro
- Definição de um limiar mínimo (min_anthill_pixels) para aceitação do patch

## 6. Tratamento de pixels ignorados

### Problema

Presença de regiões não rotuladas nas máscaras (pixels com valor 255).

### Impacto

- Introdução de ruído no cálculo da função de perda
- Gradientes inconsistentes

### Solução

Utilização do parâmetro ignore_index = 255 na função de perda, garantindo que esses pixels não contribuam para o cálculo do gradiente.

## 7. Limitações das funções de perda tradicionais

### Problema

Funções de perda tradicionais (CrossEntropy e Dice) não são adequadas para:

- Desbalanceamento extremo entre classes
- Representação precisa de bordas
- Otimização direta da métrica IoU

### Impacto

- IoU limitado (~0.30–0.35)
- Máscaras com baixa precisão espacial
- Dificuldade na detecção de regiões pequenas

### Solução

Uso de função de perda combinada:

- Tversky Loss (α=0.3, β=0.7): prioriza recall
- Lovász Loss: otimiza diretamente IoU
- Focal Loss (γ=2.0): enfatiza exemplos difíceis

```python
loss = 0.5 * Tversky + 0.3 * Lovász + 0.2 * Focal
```

## 8. Resultados das correções

| Métrica | Antes | Depois | Variação |
|---------|-------|--------|----------|
| IoU | 0.34 | 0.56 | +64.7% |
| Recall | 0.41 | 0.83 | +102% |
| Estabilidade | Baixa | Alta | — |

## 9. Considerações finais

As correções aplicadas no pipeline de dados foram determinantes para a melhoria do desempenho do modelo.

Os resultados indicam que:

- O principal gargalo estava na qualidade e distribuição dos dados, e não na arquitetura da rede
- O desbalanceamento de classes impacta diretamente o processo de otimização
- Técnicas de pré-processamento e engenharia de dados são fundamentais em tarefas de segmentação semântica

O pipeline corrigido apresentou maior estabilidade no treinamento e melhor capacidade de generalização, demonstrando a importância de um fluxo de dados bem estruturado.