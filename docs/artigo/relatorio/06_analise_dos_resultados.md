# 6. Análise dos Resultados

Este capítulo apresenta a interpretação dos resultados obtidos no treinamento do modelo de segmentação semântica para detecção de formigueiros em imagens aéreas.

---

## 1. Problema Inicial

Nos experimentos iniciais, observou-se baixo desempenho do modelo, evidenciado por métricas reduzidas de IoU e Recall.

Esse comportamento está diretamente relacionado às características do dataset:

- Predominância de pixels de fundo (~90%)
- Baixa representatividade da classe de interesse (~5–10%)

---

## 2. Impacto do Desbalanceamento

O desbalanceamento de classes influenciou diretamente o processo de aprendizado, gerando:

- Viés do modelo para a classe majoritária (fundo)
- Redução da capacidade de detecção de formigueiros
- Baixo recall
- IoU limitado (~0.35)

Nesse cenário, o modelo minimizava a função de perda prevendo majoritariamente fundo, sem aprender corretamente a classe minoritária.

---

## 3. Efeito das Estratégias Aplicadas

As técnicas introduzidas ao longo do pipeline tiveram impacto direto no comportamento do modelo.

### 3.1 Patch Training

A seleção de regiões com maior concentração de formigueiros aumentou a densidade de pixels relevantes por amostra, reduzindo o desbalanceamento local.

Como resultado:

- Melhor aprendizado da classe minoritária
- Aumento significativo do recall

---

### 3.2 Pré-processamento e consistência dos dados

A sincronização de transformações, normalização das imagens e correção das máscaras garantiram maior consistência nos dados de entrada.

Impactos observados:

- Redução de ruídos no treinamento
- Maior estabilidade no processo de otimização
- Melhor convergência da função de perda

---

### 3.3 Funções de perda especializadas

A combinação de diferentes funções de perda permitiu lidar com as limitações das abordagens tradicionais:

- **Tversky Loss (α=0.3, β=0.7):** aumentou a sensibilidade do modelo (recall)
- **Focal Loss (γ=2.0):** direcionou o aprendizado para exemplos difíceis
- **Lovász Loss:** aproximou a otimização diretamente da métrica IoU

Essa combinação reduziu o colapso para a classe de fundo e promoveu um melhor equilíbrio entre precisão e recall.

---

## 4. Resultados Obtidos

| Métrica | Antes | Depois | Variação |
|--------|------|--------|---------|
| IoU | 0.34 | 0.56 | +64.7% |
| Recall | 0.41 | 0.83 | +102% |
| Estabilidade | Baixa | Alta | — |

Os resultados indicam uma melhoria significativa na capacidade de segmentação e detecção do modelo.

---

## 5. Evolução do Treinamento

Após a aplicação das melhorias:

- Redução consistente da função de perda ao longo das épocas
- Aumento progressivo do IoU
- Melhoria na detecção de formigueiros
- Treinamento mais estável e menos sensível a variações

---

## 6. Discussão

Os resultados demonstram que o principal gargalo do problema não estava na arquitetura da rede, mas na qualidade e distribuição dos dados.

O desbalanceamento extremo levou a um comportamento degenerado do modelo, no qual prever apenas fundo era suficiente para minimizar a perda.

A combinação de técnicas de engenharia de dados (patch training) com funções de perda adaptadas foi essencial para corrigir esse comportamento.

---

## 7. Limitações

Apesar dos avanços, algumas limitações permanecem:

- Dependência da qualidade das máscaras anotadas
- Possibilidade de falsos positivos em regiões visualmente semelhantes ao formigueiro
- Baixa diversidade do dataset (cenários e condições visuais)

---

## 8. Considerações Finais

A análise dos resultados evidencia que a melhoria do pipeline de dados foi determinante para o desempenho do modelo.

Os experimentos indicam que:

- O balanceamento dos dados é crítico em segmentação semântica
- A escolha da função de perda impacta diretamente o comportamento do modelo
- Técnicas de pré-processamento são tão importantes quanto a arquitetura da rede

O modelo final demonstrou desempenho consistente, indicando viabilidade para aplicações reais em monitoramento ambiental.