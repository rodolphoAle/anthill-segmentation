# ADR-008 — Uso de U-Net para Segmentação Semântica

## Status

Aceito

---

## Contexto

Segmentação semântica de objetos pequenosMano em imagens aéreas requer:

- Extração de features multi-escala;
- Preservação de informação espacial;
- Conexão entre contexto global e detalhes locais;
- Upsampling para recuperar resolução.

Arquiteturas candidatas:

- **FCN**: Simples, mas perde informação espacial
- **DeepLab**: Complexa, custo computacional alto
- **U-Net**: Balance entre simplicity e efetividade
- **SegNet**: Boa performance, menos usada

---

## Decisão

Foi adotada **U-Net** como arquitetura principal de segmentação.

### Motivos

1. **Design simples**: Fácil implementar e debugar
2. **Skip connections**: Preservam detalhes espaciais
3. **Encoder-decoder**: Balance entre contexto e resolução
4. **Provada para formigueiros**: Literatura mostra bons resultados
5. **Menor número de parâmetros**: Treina mais rapidamente

### Arquitetura

```
Input (256x256x3)
    ↓
Encoder (downsampling)
    ↓
Bottleneck
    ↓
Decoder (upsampling + skip connections)
    ↓
Output (256x256x2)
```

### Características da Implementação

- **Encoder**: Blocos com conv + ReLU + MaxPool
- **Decoder**: Blocos com upsampling + concatenação + conv
- **Skip connections**: Concatenam features do encoder no decoder
- **Dropout**: Regularização durante treinamento
- **Batch Norm**: Normalização para convergência estável

---

## Consequências

### Positivas

 **Arquitetura provada**: U-Net é benchmark em segmentação

 **Bom trade-off**: Simplicidade vs. performance

 **Treina rapidamente**: Número moderado de parâmetros

 **Skip connections**: Preservam detalhes finos

 **Flexível**: Adaptável para diferentes tamanhos de entrada

### Negativas

 **Menos sofisticada**: Modelos recentes (Transformers) podem ser melhores

 **Dependência de dados**: Requer quantidade razoável de dados

 **Sem atenção**: Não tem mecanismos de atenção

---

## Comparação com Alternativas

| Arquitetura | Parâmetros | Tempo | Performance | Simplicidade |
| --- | --- | --- | --- | --- |
| FCN | Baixo | Rápido | Média | Alta |
| **U-Net** | **Médio** | **Rápido** | **Boa** | **Alta** |
| DeepLab | Alto | Lento | Muito Boa | Média |
| SegNet | Médio | Médio | Boa | Média |

---

## Configuração Específica

```python
unet = UNet(
    in_channels=3,      # RGB
    out_channels=2,     # Fundo + Formigueiro
    depth=4,            # Profundidade do encoder
    initial_features=64,
    bilinear=True,      # Upsampling bilinear
)
```

### Configuração Recomendada

- **Entrada**: 256x256x3 (RGB)
- **Saída**: 256x256x2 (logits para 2 classes)
- **Profundidade**: 4 (balance entre receptive field e memória)
- **Features iniciais**: 64 (escalável para datasets maiores)

---

## Implementação

**Arquivo**: `app/domain/unet.py`

**Uso**: `app/main.py`

```python
model = UNet(
    in_channels=3,
    out_channels=2,
    depth=4,
    initial_features=64,
)

training_service = TrainingService(model)
```

---

## Referências

- Ronneberger et al., 2015. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- Implementação: `app/domain/unet.py`
- Treinamento: `app/service/training_service.py`
