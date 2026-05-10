# Training Service

## Objetivo

Este arquivo implementa o `TrainingService`, responsável pelo treinamento e validação da rede U-Net utilizada na segmentação semântica de formigueiros.

O serviço centraliza toda a lógica de treinamento do modelo.

---

# Responsabilidades

O `TrainingService` é responsável por:

- executar treinamento;
- executar validação;
- calcular losses;
- atualizar pesos da rede;
- controlar epochs;
- salvar checkpoints;
- carregar modelos;
- registrar métricas e logs;
- controlar estado do treinamento.

---

# Fluxo Geral

```text
Dataset
    ↓
DataLoader
    ↓
Forward
    ↓
Loss
    ↓
Backpropagation
    ↓
Atualização dos pesos
    ↓
Validação
    ↓
Checkpoint
```

---

# Estrutura Principal

## TrainingStatus

Enum responsável pelos estados do treinamento.

| Estado      | Descrição                      |
| ----------- | ------------------------------ |
| `IDLE`      | Nenhum treinamento em execução |
| `PREPARING` | Preparando treinamento         |
| `TRAINING`  | Treinamento em andamento       |
| `COMPLETED` | Treinamento finalizado         |
| `FAILED`    | Falha durante treinamento      |

---

## TrainingState

Classe responsável por armazenar:

* estado atual;
* epoch atual;
* losses;
* mensagens de erro.

---

## TrainingService

Classe principal responsável pela execução do treinamento.

---

# Seleção do Dispositivo

O serviço seleciona automaticamente:

```text
CUDA → se disponível
CPU  → caso contrário
```

Também é possível configurar manualmente via:

```env
UNET_DEVICE=cpu
UNET_DEVICE=cuda
```

---

# Loss Utilizada

O treinamento utiliza:

* Focal Loss;
* Tversky Loss;
* Lovász Loss.

Implementadas por:

```python
CombinedTverskyFocalLoss
```

---

# Objetivo da Loss Combinada

## Focal Loss

Melhora aprendizado em datasets desbalanceados.

---

## Tversky Loss

Aumenta Recall da classe formigueiro.

---

## Lovász Loss

Melhora diretamente a métrica IoU.

---

# Otimizador

O treinamento utiliza:

```python
Adam
```

---

# Scheduler

Scheduler utilizado:

```python
ReduceLROnPlateau
```

Objetivo:
reduzir learning rate automaticamente quando a validação estagna.

---

# AMP (Mixed Precision)

O AMP foi desabilitado:

```python
use_amp = False
```

Motivo:
foram observadas instabilidades numéricas na U-Net,
principalmente no bottleneck com muitos canais.

---

# Validações de Segurança

O treinamento ignora batches inválidos:

## Entradas inválidas

```python
NaN
Inf
```

---

## Labels inválidas

```python
labels == 255
```

---

## Outputs inválidos

```python
NaN
Inf
```

---

## Loss inválida

```python
NaN
Inf
```

---

# Clipping de Gradiente

O treinamento utiliza:

```python
clip_grad_norm_
```

Objetivo:
evitar explosão de gradientes.

---

# Salvamento do Melhor Modelo

O serviço salva automaticamente:

```python
best_model_params.pth
```

quando ocorre melhora na loss de validação.

---

# Loop de Treinamento

O método principal:

```python
_train_loop()
```

executa:

1. forward;
2. cálculo da loss;
3. backpropagation;
4. atualização dos pesos;
5. logs;
6. validação;
7. checkpoint.

---

# Loop de Validação

O método:

```python
_evaluate_loop_sync()
```

executa validação sem cálculo de gradiente.

---

# Execução Assíncrona

Treinamentos são executados usando:

```python
asyncio.to_thread()
```

Objetivo:
evitar bloqueio do loop async principal.

---

# Principais Métodos

| Método                | Responsabilidade       |
| --------------------- | ---------------------- |
| `_resolve_device`     | Seleciona CPU/GPU      |
| `_train_loop`         | Executa treinamento    |
| `_evaluate_loop_sync` | Executa validação      |
| `start_training`      | Inicializa treinamento |
| `save_model`          | Salva pesos            |
| `load_model`          | Carrega pesos          |

---

# Entrada e Saída

## Entrada

| Entrada        | Descrição          |
| -------------- | ------------------ |
| `train_loader` | Dados de treino    |
| `val_loader`   | Dados de validação |
| `criterion`    | Função de perda    |
| `optimizer`    | Otimizador         |

---

## Saída

| Saída           | Descrição            |
| --------------- | -------------------- |
| Modelo treinado | Rede atualizada      |
| Checkpoints     | Pesos salvos         |
| Logs            | Métricas e progresso |

---

# Importância no Projeto

O `TrainingService` é o núcleo do pipeline de treinamento.

Ele garante:

* estabilidade;
* controle de execução;
* persistência do modelo;
* validação contínua;
* rastreabilidade do treinamento.

---

# Relação com Outros Arquivos

| Arquivo                   | Relação                      |
| ------------------------- | ---------------------------- |
| `segmentation_dataset.py` | Fonte de dados local         |
| `streaming_dataset.py`    | Fonte de dados remota        |
| `combined_loss.py`        | Loss principal               |
| `metrics.py`              | Avaliação do modelo          |
| `config.py`               | Configurações do treinamento |
| `unet.py`                 | Arquitetura da rede          |

---

# Resumo

O `training_service.py` centraliza todo o treinamento da U-Net.

Ele executa:

* treinamento;
* validação;
* controle de estado;
* cálculo de losses;
* atualização dos pesos;
* salvamento automático do melhor modelo.

Esse serviço foi fundamental para permitir treinamento estável em um dataset altamente desbalanceado.
