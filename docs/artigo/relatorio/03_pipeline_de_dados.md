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

- Augmentação de duplicação de formigueiros (intra-tile): cópias rotacionadas/espelhadas dos formigueiros já presentes no recorte (p=0,7, até 2 cópias), aumentando a proporção de pixels positivos sem coletar novos dados
- Função de custo combinada (Tversky β=0,7 + Focal + Lovász) com peso maior para a classe formigueiro (6,0)

## 5. Tiles dominados por área não anotada

### Problema

Muitos recortes nas bordas dos talhões têm grande parte da máscara como "ignorar" (não anotada), oferecendo pouco sinal de aprendizado.

### Impacto

- Treinamento ineficiente (gradiente desperdiçado)
- Reforço do viés para fundo

### Solução

- Descarte dos recortes cuja máscara tem mais de 70% de pixels "ignorar" (`max_ignore_pixel_pct = 0.7`)

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