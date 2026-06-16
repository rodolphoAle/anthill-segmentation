# Architecture Decision Records (ADRs)

Este diretório contém decisões arquiteturais documentadas seguindo o padrão ADR.

## O que é um ADR?

Um **Architecture Decision Record** é um documento que captura uma decisão arquitetural importante, seu contexto, consequências e justificativas.

**Estrutura padrão**:

1. **Título**: `ADR-XXX — Descrição da Decisão`
2. **Status**: Aceito, Em análise, Obsoleto ou Substituído
3. **Contexto**: Por que a decisão foi necessária
4. **Decisão**: O que foi decidido e por quê
5. **Consequências**: Impactos positivos e negativos

---

## ADRs Atuais

| ID | Título | Status | Área |
|---|---|---|---|
| [ADR-001](ADR-001-focal-loss.md) | Uso de Focal Loss | Aceito | Loss |
| [ADR-002](ADR-002-tversky-loss.md) | Uso de Tversky Loss | Aceito | Loss |
| [ADR-003](ADR-003-lovasz-loss.md) | Uso de Lovász Hinge Loss | Aceito | Loss |
| [ADR-004](ADR-004-combined-loss.md) | Loss Combinada | Aceito | Loss |
| [ADR-005](ADR-005-streaming-dataset.md) | Streaming Dataset | Aceito | Data |
| [ADR-006](ADR-006-region-filter.md) | Filtragem de Regiões | Aceito | Pós-processamento |
| [ADR-007](ADR-007-async-training.md) | Treinamento Assíncrono | Aceito | Arquitetura |
| [ADR-008](ADR-008-unet-architecture.md) | Arquitetura U-Net | Aceito | Modelo |
| [ADR-009](ADR-009-imagenet-normalization.md) | Normalização ImageNet | Aceito | Pré-processamento |

---

## Por que ADRs?

 **Rastreabilidade**: Justifica decisões técnicas

 **Onboarding**: Novos desenvolvedores entendem racional

 **Profissionalismo**: Padrão da engenharia de software

 **Documentação**: Preserva conhecimento do projeto

 **Referência**: Facilita revisão de decisões

---

## Como Adicionar um Novo ADR

1. Crie um arquivo `ADR-NXX-titulo-descritivo.md`
2. Siga o padrão de estrutura
3. Use número sequencial (próximo ID disponível)
4. Atualize este README

Exemplo:

```markdown
# ADR-010 — Descrição da Próxima Decisão

## Status
Aceito

## Contexto
Explique o problema ou motivação...

## Decisão
Explique o que foi decidido...

## Consequências
### Positivas
- Benefício 1
- Benefício 2

### Negativas
- Desvantagem 1
- Desvantagem 2
```

---

## Relação com Documentação Técnica

- **ADRs**: *Por que* decidir assim (justificativa)
- **Technical Docs**: *Como* implementar (funcionamento)

Juntas, formam documentação completa do projeto.

---

## Referências

- [ADR - Lightweight ADR web framework](https://adr.github.io/)
- [Architectural Decision Records](https://thinkrelevant.com/blog/architectural-decision-records/)
