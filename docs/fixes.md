# Correções no Pipeline de Dados

Este documento descreve os principais problemas identificados no pipeline de dados e as respectivas soluções implementadas.

---

## 1. Transformações desalinhadas

### Problema
As transformações (rotação, flip, entre outras) eram aplicadas separadamente na imagem e na máscara.

### Impacto
- Desalinhamento entre imagem e rótulo
- Comprometimento do aprendizado supervisionado
- Baixo desempenho (IoU não evoluía)

### Solução

Aplicação sincronizada das transformações:

```python
image, mask = transforms(image, mask)
```

## 2. Ausência de normalização

### Problema
As imagens eram utilizadas sem normalização.

### Impacto
- Entrada de dados inconsistente
- Instabilidade no treinamento
- Convergência mais lenta

### Solução

Aplicação de normalização padrão:

```python
image = TF.normalize(image, mean, std)
```

## 3. Valores inválidos nas máscaras

### Problema
Após as transformações, as máscaras apresentavam valores inconsistentes (diferentes de 0, 1 ou 255).

### Impacto
- Erros no cálculo da função de perda
- Introdução de ruído no treinamento

### Solução

Padronização dos valores das máscaras:

```python
label = np.where(label > 1, 255, label)
```
## 4. Desbalanceamento de classes

### Problema
Predominância de pixels de fundo no dataset.

### Impacto
- Modelo enviesado para a classe majoritária
- Baixo recall
- IoU limitado (~0.35)

### Solução

Aplicação da técnica de Patch Training, com seleção de regiões contendo maior proporção de formigueiros.

---

## 5. Patches sem informação relevante

### Problema
Alguns recortes não continham pixels da classe de interesse.

### Impacto
- Treinamento ineficiente
- Persistência do viés do modelo

### Solução

- Realização de múltiplas tentativas de extração
- Seleção do patch com maior proporção de formigueiros

---

## 6. Resultados das correções

| Métrica | Antes | Depois |
|--------|------|--------|
| IoU | ~0.35 | ~0.55+ |
| Recall | Baixo | Alto |
| Estabilidade | Baixa | Alta |

---

## 7. Considerações finais

As correções aplicadas no pipeline de dados foram determinantes para a melhoria do desempenho do modelo.

Observou-se que problemas relacionados à qualidade e preparação dos dados tiveram impacto mais significativo do que alterações na arquitetura da rede.