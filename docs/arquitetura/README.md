# Arquitetura do Projeto

Documentação completa da arquitetura do TCC-UFMS, dividida em dois formatos profissionais:

## ADRs (Architecture Decision Records)

Decisões arquiteturais importantes documentadas formalmente.

**Pasta**: `./adr/`

Útil para:
- Entender **por quê** as decisões foram tomadas
- Justificar escolhas técnicas
- Onboarding de novos desenvolvedores
- Avaliação academica

**Leitura recomendada**: Comece pelos ADRs para entender as decisões principais.

---

##  Documentação Técnica

Detalhes de implementação e funcionamento.

**Pasta**: `./technical/`

Útil para:
- Entender **como** os componentes funcionam
- Referência durante desenvolvimento
- Debugging de problemas
- Uso das APIs

**Leitura recomendada**: Consulte documentação técnica ao integrar ou modificar componentes.

---

##  Estrutura Completa

```
arquitetura/
├── README.md (este arquivo)
├── adr/
│   ├── README.md (guia de ADRs)
│   ├── ADR-001-focal-loss.md
│   ├── ADR-002-tversky-loss.md
│   ├── ADR-003-lovasz-loss.md
│   ├── ADR-004-combined-loss.md
│   ├── ADR-005-streaming-dataset.md
│   ├── ADR-006-region-filter.md
│   ├── ADR-007-async-training.md
│   ├── ADR-008-unet-architecture.md
│   └── ADR-009-imagenet-normalization.md
│
├── technical/
│   ├── combined-loss.md
│   ├── lovasz-loss.md
│   ├── segmentation-dataset.md
│   ├── prediction-service.md
│   └── training-service.md
│
├── domain/ (histórico)
├── infrastructure/ (histórico)
└── service/ (histórico)
```

---

##  Guia de Leitura Recomendado

### Para Apresentação ao Professor

1. Leia [ADRs README](adr/README.md) para entender formato
2. Leia todos os 9 ADRs (10-15 min cada)
3. Consulte technical docs conforme necessário para detalhar implementação

### Para Desenvolvimento

1. Consulte ADRs para entender contexto e justificativa
2. Use technical docs para implementar
3. Siga padrões documentados

### Para Debugging

1. Procure na documentação técnica correspondente
2. Consulte ADR para entender design e trade-offs
3. Modifique seguindo padrões existentes

---

##  ADRs Disponíveis

| ID | Título | Status | Tema |
|---|---|---|---|
| [ADR-001](adr/ADR-001-focal-loss.md) | Uso de Focal Loss |  Aceito | Tratamento de desbalanceamento |
| [ADR-002](adr/ADR-002-tversky-loss.md) | Uso de Tversky Loss |  Aceito | Maximização de Recall |
| [ADR-003](adr/ADR-003-lovasz-loss.md) | Uso de Lovász Loss |  Aceito | Otimização de IoU |
| [ADR-004](adr/ADR-004-combined-loss.md) | Loss Combinada |  Aceito | Estratégia de treinamento |
| [ADR-005](adr/ADR-005-streaming-dataset.md) | Streaming Dataset |  Aceito | Escalabilidade de dados |
| [ADR-006](adr/ADR-006-region-filter.md) | Filtragem de Regiões |  Aceito | Pós-processamento |
| [ADR-007](adr/ADR-007-async-training.md) | Treinamento Assíncrono |  Aceito | Arquitetura de serviço |
| [ADR-008](adr/ADR-008-unet-architecture.md) | Arquitetura U-Net |  Aceito | Escolha de modelo |
| [ADR-009](adr/ADR-009-imagenet-normalization.md) | Normalização ImageNet |  Aceito | Pré-processamento |

---

##  Relação com Código

Cada documento é linkado aos arquivos de implementação:

```
ADR-001 (decisão arquitetural)
    ↓
app/domain/losses/focal_loss.py (implementação)
    ↓
technical/combined-loss.md (detalhe técnico)
```

---

##  Estrutura de Componentes

### Losses (Decisões 1-4)
-  **ADR-001**: Por quê Focal Loss?
-  **ADR-002**: Por quê Tversky Loss?
-  **ADR-003**: Por quê Lovász Loss?
-  **ADR-004**: Como combinar?
-  **Technical**: Implementação completa

### Data & Escalabilidade (Decisão 5 + 9)
-  **ADR-005**: Streaming Dataset
-  **ADR-009**: Normalização ImageNet
-  **Technical**: Segmentation Dataset

### Modelo & Treinamento (Decisões 7-8)
-  **ADR-007**: Arquitetura assíncrona
-  **ADR-008**: Arquitetura U-Net
-  **Technical**: Training & Prediction Services

### Pós-processamento (Decisão 6)
-  **ADR-006**: Filtragem de regiões
-  **Technical**: Prediction Service details

---

##  Padrão Utilizado

Esta estrutura segue **padrões industriais** de:
- Architecture Decision Records (ADRs)
- Software Engineering Best Practices
- Machine Learning Documentation
- Professional Technical Writing


---

## Próximos Passos para Novos Componentes

Se adicionar novo componente importante:

1. Criar ADR justificando a decisão (contexto, decisão, consequências)
2. Implementar no código seguindo padrões
3. Documentar tecnicamente em `technical/`
4. Atualizar este README
5. Linkar ADR → Código → Documentation

## infrastructure/

Contém datasets e acesso aos dados.

| Arquivo                   | Responsabilidade            |
| ------------------------- | --------------------------- |
| `segmentation_dataset.md` | Dataset local               |
| `streaming_dataset.md`    | Dataset remoto em streaming |

---

## service/

Contém serviços principais do pipeline.

| Arquivo                 | Responsabilidade |
| ----------------------- | ---------------- |
| `prediction_service.md` | Inferência       |
| `training_service.md`   | Treinamento      |
| `validation_service.md` | Validação        |

---

# Fluxo Geral

```text
Dataset
    ↓
Augmentations
    ↓
U-Net
    ↓
Loss
    ↓
Treinamento
    ↓
Validação
    ↓
Métricas
```

---

# Modelo Utilizado

O projeto utiliza:

```text
U-Net
```

para segmentação semântica binária:

* fundo;
* formigueiro.

---

# Principais Técnicas

| Técnica           | Objetivo                                   |
| ----------------- | ------------------------------------------ |
| Focal Loss        | Melhorar aprendizado da classe minoritária |
| Tversky Loss      | Melhorar Recall                            |
| Lovász Loss       | Melhorar IoU                               |
| Streaming Dataset | Reduzir uso de disco                       |
| Region Filter     | Remover ruídos                             |

---

# Métricas Utilizadas

| Métrica        | Objetivo        |
| -------------- | --------------- |
| Pixel Accuracy | Acurácia global |
| IoU            | Sobreposição    |
| Dice Score     | Similaridade    |
| Precision      | Precisão        |
| Recall         | Sensibilidade   |
| F1 Score       | Equilíbrio      |

---

# Objetivo da Documentação

Esta documentação foi criada para:

* facilitar manutenção;
* auxiliar apresentação do artigo;
* explicar arquitetura do sistema;
* documentar funcionamento do pipeline.

