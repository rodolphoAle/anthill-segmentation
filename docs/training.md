# Training

Execução do treinamento do modelo U-Net para segmentação de formigueiros.

---

## Setup

### Pré-requisitos

| Item | Requisito |
|------|----------|
| Python | 3.8+ |
| Framework | PyTorch |
| Dependências | requirements.txt |

### Instalação

```pip install -r requirements.txt``` 

### Dataset 

| Item           | Descrição |
| -------------- | --------- |
| Diretório base | `data/`   |
| Imagens        | `rgb/`    |
| Máscaras       | `labels/` |


## Estrutura 

data/
├── rgb/
└── labels/

## Verificações

Antes do treinamento, validar ambiente e dados:

```python setup_training.py --check```
```python setup_training.py --check-dataset```
```python validate_fixes.py```

### Resultado esperado
Ambiente OK
Dataset encontrado
Validações concluídas com sucesso

## Execução do Treinamento

Opção 1  Automatizada (recomendada)
```bash quick_start.sh```

Opção 2  Manual
```python train_with_monitoring.py \
  --epochs 5 \
  --lr 1e-4 \
  --batch-size 4 \
  --local-data \
  --data-dir ./data/
```
## Parâmetros
Parâmetro	Descrição
--epochs	Número de épocas
--lr	Taxa de aprendizado
--batch-size	Tamanho do batch
--data-dir	Caminho do dataset

## Função de Perda
O treinamento utiliza uma função de perda composta: 
```Total Loss = 0.5 * Tversky + 0.3 * Lovász + 0.2 * Focal```
### Componentes
#### Tversky Loss
Controla o equilíbrio entre FP e FN
Configuração:
alpha = 0.4
beta = 0.6
#### Focal Loss
Reduz o impacto de exemplos fáceis (fundo dominante)
#### Lovász Hinge Loss
Otimiza diretamente o IoU 
Melhora contornos e segmentação fina das imagens de formigueiro

#### Objetivo 
Melhorar recall (detecção)
Melhorar IoU (qualidade da máscara)
Reduzir impacto do desbalanceamento de classes

## Saída

Estrutura gerada
training_output/
├── training_metrics_*.csv
└── predictions_epoch*/

## Resultados Esperados
| Época | IoU esperado |
| ----- | ------------ |
| 1     | ~0.40        |
| 3     | ~0.50        |
| 5     | ~0.55+       |


## Problemas Comuns
### IoU não evolui

### Causas:

Dataset inconsistente
Máscaras incorretas
Desbalanceamento extremo

### Solução:

```python validate_fixes.py```

### Segmentação ruim (máscaras imprecisas)
### Causas
Bordas mal definidas
Alta proporção de fundo

### Solução:
Uso da loss combinada (Tversky + Focal + Lovász)
Aumento da qualidade dos dados

### Treinamento lento

### Causas:

Uso de CPU
Batch alto

### Soluções:

Utilizar GPU
Reduzir batch-size

### Erro no dataset

### Causa:

Estrutura incorreta

#### Solução:
Verificar diretórios:

data/
├── rgb/
└── labels/

Considerações finais

Se configurado corretamente, o modelo converge para valores de IoU superiores a 0.55 em poucas épocas, apresentando:

Alta capacidade de detecção, refletida em valores elevados de recall, indicando que a maioria dos formigueiros é corretamente identificada;
Evolução consistente do IoU ao longo das épocas, evidenciando melhoria na qualidade da segmentação;
Maior precisão na delimitação das regiões de interesse, decorrente do uso da função de perda combinada (Focal + Tversky + Lovász), que equilibra desbalanceamento de classes e otimiza diretamente a métrica de sobreposição.