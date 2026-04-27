# Training

Execução do treinamento do modelo U-Net para segmentação de formigueiros.
---

## Setup
### Pré-requisitos
Item	Requisito
Python	3.8+
Framework	PyTorch instalado
Dependências	requirements.txt

## Instalação:

pip install -r requirements.txt
Dataset
Item	Descrição
Diretório base	data/
Imagens	rgb/
Máscaras	labels/

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

## Saída

Estrutura gerada
training_output/
├── training_metrics_*.csv
└── predictions_epoch*/

## Resultados Esperados
Época	IoU esperado
1	~0.40
3	~0.55
5	~0.55+

## Problemas Comuns
### IoU não evolui

### Causas:

Dataset inconsistente
Máscaras incorretas

### Solução:

```python validate_fixes.py```

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

Se configurado corretamente, o modelo deve convergir para IoU superior a 0.55 em poucas épocas.