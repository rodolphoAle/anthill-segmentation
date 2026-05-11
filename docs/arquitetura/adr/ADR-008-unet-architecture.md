# ADR-008 — Uso de U-Net para Segmentação Semântica

## Status

Aceito

---

## Contexto

Segmentação semântica de objetos pequenos em imagens aéreas requer:

- Extração de características em múltiplas escalas;
- Preservação de informação espacial através da arquitetura;
- Conexão entre contexto global e detalhes locais;
- Reconstrução espacial para recuperar resolução original.

Arquiteturas candidatas analisadas:

- **FCN (Fully Convolutional Networks)**: Arquitetura simples, porém com perda significativa de informação espacial durante pooling
- **DeepLab**: Atrous convolutions e CRF otimizam IoU, com custo computacional elevado
- **U-Net**: Balanceamento favorável entre custo computacional e performance em segmentação
- **SegNet**: Performance comparável, porém menos documentada na literatura de objetos pequenos

---

## Decisão

Foi adotada **U-Net** como arquitetura principal de segmentação semântica.

### Justificativa Técnica

1. **Custo computacional reduzido**: Arquitetura com ~7.8M parâmetros (vs. 38M em DeepLab V3), permitindo treinamento com datasets limitados
2. **Preservação de características espaciais**: Skip connections concatenam features do encoder no decoder, essencial para segmentação precisa de objetos pequenos
3. **Balanceamento encoder-decoder**: Profundidade adequada (5 níveis) para manter receptive field sem excessiva redução de resolução
4. **Amplamente validada em literatura**: Ronneberger et al. (2015) demonstraram performance superior em segmentação de objetos pequenos
5. **Flexibilidade arquitetural**: Permite inserção de normalização, múltiplos upsampling strategies e regularização sem redesign completo

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

- **Encoder (downsampling)**: Blocos sequenciais de convolução 3×3 + ReLU + pooling 2×2, extraindo características em escalas progressivas
- **Decoder (upsampling)**: Upsampling 2×2 (bilinear ou ConvTranspose2d) + concatenação com skip connections + convolução dupla
- **Skip connections**: Concatenam features de alta resolução do encoder no decoder correspondente, preservando detalhes finos
- **Batch Normalization**: Normaliza ativações por batch, estabilizando gradientes e acelerando convergência (adicionado em iterações posteriores)
- **Regularização**: Dropout opcional para reduzir overfitting em datasets pequenos

---

## Consequências

### Consequências Positivas

✓ **Amplamente validada academicamente**: Mais de 10.000 citações; estabelecida como baseline em segmentação semântica desde 2015

✓ **Custo computacional controlado**: Convergência em 50-100 épocas com GPU padrão, viável para equipes com recursos limitados

✓ **Performance em objetos pequenos**: Skip connections demonstraram preservação superior de bordas comparado a FCN

✓ **Implementação transparente**: Arquitetura simples facilita debugging e ablation studies

✓ **Adaptabilidade**: Profundidade, número de filtros iniciais e estratégias de upsampling são facilmente modificáveis

### Consequências Negativas

✗ **Capacidade representacional limitada**: Sem mecanismos de atenção, não captura dependências de longo alcance como Vision Transformers

✗ **Sensibilidade ao balanceamento de dados**: Desempenho significativamente reduzido com datasets com forte desbalanceamento de classes

✗ **Arquitetura convolucional**: Receptive field limitado comparado a abordagens com dilatação (DeepLab) ou arquiteturas baseadas em transformers

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

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, pp. 234–241. Springer.
- Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. In *CVPR*, pp. 3431–3440.
- Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *TPAMI*, 40(4), 834–848.

## Implementação

- Código: [app/domain/unet.py](../../app/domain/unet.py)
- Treinamento: [app/service/training_service.py](../../app/service/training_service.py)
- Experimentos: [docs/artigo/relatorio/04_experimentos.md](../artigo/relatorio/04_experimentos.md)
