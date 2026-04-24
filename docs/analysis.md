# Análise Técnica dos Resultados

Este documento apresenta a análise dos resultados obtidos no treinamento do modelo de segmentação semântica para detecção de formigueiros em imagens aéreas.

---

## 1. Problema Inicial

Durante os experimentos iniciais, observou-se baixo desempenho do modelo, principalmente devido ao desbalanceamento de classes no dataset.

### Características do dataset

- Predominância de pixels de fundo (~90%)
- Baixa representatividade de formigueiros (~5–10%)

---

## 2. Impacto no Modelo

O desbalanceamento gerou os seguintes efeitos:

- Tendência do modelo em prever apenas a classe majoritária (fundo)
- Baixo recall (falha na detecção de formigueiros)
- IoU limitado (~0.35)

---

## 3. Estratégias de Mitigação

Foram aplicadas técnicas para melhorar a representatividade dos dados e a qualidade do treinamento:

### 3.1 Patch Training
Uso de recortes das imagens com maior concentração de formigueiros.

### 3.2 Transformações sincronizadas
Aplicação conjunta de augmentations em imagem e máscara.

### 3.3 Normalização
Padronização dos dados de entrada com estatísticas do ImageNet.

### 3.4 Correção das máscaras
Garantia de valores válidos para treinamento supervisionado.

---

## 4. Resultados

| Métrica | Antes | Depois |
|--------|------|--------|
| IoU | ~0.35 | ~0.55+ |
| Recall | Baixo | Alto |
| Estabilidade | Baixa | Alta |

---

## 5. Evolução do Treinamento

Após as melhorias:

- Redução consistente da função de perda
- Aumento progressivo do IoU
- Maior capacidade de detecção de formigueiros

---

## 6. Discussão

Os resultados indicam que:

- O principal gargalo estava no dataset, não na arquitetura do modelo
- O desbalanceamento de classes impacta diretamente o aprendizado
- Técnicas de pré-processamento são fundamentais em tarefas de segmentação

---

## 7. Limitações

- Dependência da qualidade das máscaras
- Possibilidade de falsos positivos em regiões visualmente semelhantes
- Dataset com baixa diversidade de cenários

---

## 8. Conclusão

A aplicação de técnicas de balanceamento e pré-processamento, especialmente o patch training, foi determinante para a melhoria do desempenho do modelo.

O sistema demonstrou capacidade consistente de detecção, indicando viabilidade para aplicações em monitoramento ambiental.