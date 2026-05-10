# Lovász Hinge Loss

## Objetivo

Este arquivo implementa a `LovaszHingeLoss`, uma função de perda usada em segmentação semântica binária.

O objetivo principal é otimizar diretamente a métrica IoU, que mede a sobreposição entre a máscara prevista pelo modelo e a máscara real.

---

## Problema Resolvido

Losses como CrossEntropy, Focal Loss e Tversky Loss trabalham principalmente com erro por pixel.

A Lovász Hinge Loss é usada para aproximar melhor a otimização do IoU, tornando o treinamento mais alinhado com a métrica usada na avaliação.

---

## Estratégia

Para segmentação binária, os logits são convertidos em uma margem:

```text
margin = logit_formigueiro - logit_fundo
```

Essa margem indica o quanto o modelo favorece a classe formigueiro em relação ao fundo.

---

## Principais Parâmetros

| Parâmetro      | Descrição                                    |
| -------------- | -------------------------------------------- |
| `ignore_index` | Valor da máscara ignorado no cálculo da loss |

---

## Fluxo de Execução

1. Calcula a margem entre formigueiro e fundo.
2. Remove pixels marcados com `ignore_index`.
3. Converte o alvo para formato binário.
4. Calcula os erros da margem.
5. Ordena os erros do maior para o menor.
6. Calcula o gradiente da extensão Lovász.
7. Retorna a loss final.

---

## Entrada e Saída

| Entrada   | Tipo           | Descrição                            |
| --------- | -------------- | ------------------------------------ |
| `inputs`  | `torch.Tensor` | Logits produzidos pelo modelo        |
| `targets` | `torch.Tensor` | Máscara real com classes 0, 1 ou 255 |

| Saída | Tipo           | Descrição                          |
| ----- | -------------- | ---------------------------------- |
| Loss  | `torch.Tensor` | Valor escalar usado no treinamento |

---

## Importância no Projeto

A `LovaszHingeLoss` é importante porque aproxima o treinamento da métrica IoU.

Isso ajuda o modelo a produzir máscaras com melhor sobreposição em relação às máscaras reais.

---

## Relação com Outros Arquivos

| Arquivo               | Relação                                         |
| --------------------- | ----------------------------------------------- |
| `combined_loss.py`    | Pode usar a Lovász como parte da loss combinada |
| `metrics.py`          | Calcula IoU e Dice para avaliar o modelo        |
| `training_service.py` | Usa a loss durante o treinamento                |

---

## Resumo 

A Lovász Hinge Loss foi adicionada para melhorar a segmentação considerando diretamente o IoU.

Enquanto outras losses ajudam no aprendizado por pixel, a Lovász aproxima o treinamento da métrica final usada para avaliar a qualidade da máscara.
