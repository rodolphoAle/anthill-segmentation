# 3. Pipeline de Dados

Este documento descreve os principais problemas identificados no pipeline de dados para segmentação semântica de formigueiros, bem como as soluções implementadas.

---

## 1. Transformações desalinhadas

### Problema

As transformações geométricas eram aplicadas de forma independente na imagem e na máscara.

### Impacto

- Desalinhamento entre imagem e rótulo
- Violação da correspondência pixel a pixel
- Comprometimento do aprendizado supervisionado

### Solução

Aplicação sincronizada das transformações:

```python
image, mask = transforms(image, mask)
```

## 2. Ausência de normalização

### Problema

As imagens eram utilizadas sem padronização dos valores de entrada.

### Impacto

- Distribuição inconsistente dos dados
- Instabilidade no treinamento
- Convergência mais lenta

### Solução

Aplicação de normalização padrão:

```python
image = TF.normalize(image, mean, std)
```

## 3. Valores inválidos nas máscaras

### Problema

As máscaras apresentavam valores inconsistentes após transformações.

### Impacto

- Erros no cálculo da função de perda
- Introdução de ruído no gradiente

### Solução

Padronização dos valores:

```python
label = np.where(label > 1, 255, label)
```

## 4. Desbalanceamento de classes

### Problema

Predominância da classe fundo no dataset.

### Impacto

- Viés do modelo
- Baixo recall
- IoU limitado

### Solução

Aplicação de Patch Training:

- Extração de patches (512×512)
- Priorização de regiões com formigueiros

## 5. Patches sem informação relevante

### Problema

Patches sem presença da classe de interesse.

### Impacto

- Treinamento ineficiente
- Reforço do viés para fundo

### Solução

- Múltiplas tentativas de extração
- Seleção baseada em quantidade mínima de pixels positivos

## 6. Tratamento de pixels ignorados

### Problema

Presença de regiões não rotuladas (valor 255).

### Impacto

- Ruído no cálculo da loss
- Gradientes inconsistentes

### Solução

Uso de `ignore_index = 255` na função de perda.

## 7. Limitações das funções de perda tradicionais

### Problema

CrossEntropy e Dice não lidam bem com desbalanceamento extremo.

### Impacto

- Baixo desempenho em regiões pequenas
- IoU limitado

### Solução

Uso de funções de perda combinadas:

- Tversky Loss
- Focal Loss
- Lovász Loss

## 8. Considerações finais

As melhorias no pipeline de dados foram essenciais para viabilizar o aprendizado do modelo.

As principais contribuições incluem:

- Correção de inconsistências nos dados
- Redução do desbalanceamento
- Melhoria na qualidade das amostras de treino

Essas etapas foram determinantes para a evolução do desempenho observada nos experimentos.