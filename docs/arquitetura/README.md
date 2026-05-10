# Arquitetura do Projeto

## Objetivo

Esta pasta contém a documentação técnica da arquitetura do projeto de segmentação semântica de formigueiros utilizando U-Net.

---

# Estrutura

```text
docs/arquitetura/
├── README.md
├── domain/
│   ├── losses/
│   │   ├── combined_loss.md
│   │   ├── focal_loss.md
│   │   ├── lovasz_loss.md
│   │   └── tversky_loss.md
│   ├── mask_utils.md
│   └── metrics.md
│
├── infrastructure/
│   ├── segmentation_dataset.md
│   └── streaming_dataset.md
│
└── service/
    ├── prediction_service.md
    ├── training_service.md
    └── validation_service.md
````

---

# Organização das Pastas

## domain/

Contém regras de negócio e componentes centrais do treinamento.

| Arquivo            | Responsabilidade      |
| ------------------ | --------------------- |
| `combined_loss.md` | Loss combinada        |
| `focal_loss.md`    | Focal Loss            |
| `lovasz_loss.md`   | Lovász Loss           |
| `tversky_loss.md`  | Tversky Loss          |
| `mask_utils.md`    | Conversão de máscaras |
| `metrics.md`       | Métricas de avaliação |

---

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

