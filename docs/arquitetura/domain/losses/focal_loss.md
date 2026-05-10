# Focal Loss

## Objetivo

Reduzir o impacto de exemplos fáceis e aumentar o foco
em regiões difíceis de segmentar.

---

## Problema Resolvido

Em datasets desbalanceados, o modelo tende a prever apenas fundo.

A Focal Loss aumenta a penalização de erros difíceis,
melhorando o aprendizado da classe minoritária.

---

## Fórmula

FL(pt) = -(1 - pt)^γ * log(pt)

---

## Principais Parâmetros

| Parâmetro | Descrição |
|---|---|
| gamma | Intensidade do foco em exemplos difíceis |
| weight | Peso por classe |
| ignore_index | Pixels ignorados |

---

## Fluxo de Execução

1. Calcula CrossEntropy por pixel
2. Calcula probabilidade da predição correta
3. Aplica fator focal
4. Retorna média da loss

---

## Benefícios

- Melhor aprendizado em datasets desbalanceados
- Redução de viés para fundo
- Melhor detecção de formigueiros