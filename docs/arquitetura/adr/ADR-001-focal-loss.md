# ADR-001 — Uso de Focal Loss para Lidar com Desbalanceamento de Classes

## Status

Aceito

---

## Contexto

O dataset de treinamento possui forte desbalanceamento entre as classes:

- **Fundo**: maioria dos pixels (>90%);
- **Formigueiro**: minoria dos pixels (<10%).

Testes iniciais com CrossEntropy tradicional revelaram que o modelo:

- tendia a prever majoritariamente fundo;
- ignorava completamente regiões de formigueiro;
- atingia alta acurácia por bias (sempre predizer a classe majoritária).

Este desbalanceamento causava colapso do modelo para a classe majoritária, impedindo detecção efetiva de formigueiros.

---

## Decisão

Foi adotada **Focal Loss** como componente principal da loss combinada.

### Características da Focal Loss

A Focal Loss funciona através de:

1. **Redução do peso de exemplos fáceis**: Pixels de fundo são geralmente fáceis de classificar
2. **Aumento do foco em exemplos difíceis**: Pixels de formigueiro recebem maior peso
3. **Fator de foco (gamma)**: Controla intensidade do enfoque (gamma=2.0 no projeto)

### Objetivo

- Reduzir influência excessiva da classe majoritária;
- Aumentar gradientes para pixels misclassificados;
- Forçar modelo a aprender melhor a classe minoritária.

---

## Consequências

### Positivas

 **Melhor aprendizado do formigueiro**: Modelo passa a detectar formigueiros com maior Recall

 **Maior estabilidade inicial**: Evita colapso para fundo nos primeiros epochs

 **Redução de viés**: Loss não é dominada pela classe majoritária

 **Improved Recall**: Métrica importante para detecção é significativamente melhorada

### Negativas

 **Sensibilidade ao hiperparâmetro gamma**: Ajuste fino necessário

 **Maior custo computacional**: Cálculo mais complexo que CrossEntropy

 **Requer tuning adicional**: Necessário equilibrar gamma com outros pesos de loss

---

## Referências

- Lin et al., 2017. "Focal Loss for Dense Object Detection"
- Implementação: `app/domain/losses/focal_loss.py`
- Uso: `app/domain/losses/combined_loss.py`
