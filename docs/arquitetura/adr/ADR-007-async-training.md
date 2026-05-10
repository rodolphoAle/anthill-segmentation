# ADR-007 — Arquitetura de Serviço de Treinamento Assíncrono

## Status

Aceito

---

## Contexto

Treinamento de rede neural é computacionalmente intensivo:

- Forward pass: processamento em lote
- Backward pass: cálculo de gradientes
- Atualização: otimizador modifica pesos

Se executado na thread principal:

-  API fica travada;
-  Usuário não consegue consultar estado;
-  Impossível processar outras requisições;
-  Experiência ruim em produção.

Necessário: **desacoplar treinamento da API**.

---

## Decisão

Foi implementado **TrainingService com suporte assíncrono**:

```python
async def start_training(train_loader, val_loader, ...):
    # Executa treinamento em thread separada
    await asyncio.to_thread(
        self._train_loop,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs,
        scaler,
    )
```

### Arquitetura

```
API (async)
    ↓
start_training() 
    ↓
asyncio.to_thread()
    ↓
Thread Pool
    ↓
_train_loop() [CPU/GPU bound]
    ↓
Retorna ao evento loop
```

### Benefícios

- API permanece responsiva
- Múltiplas requisições processadas
- Estado consultável via `/training/status`

---

## Consequências

### Positivas

 **API responsiva**: Treinamento não bloqueia servidor

 **Escalabilidade**: Suporta múltiplas requisições simultâneas

 **Monitoramento**: Estado consultável em tempo real

 **Produção-ready**: Padrão moderno de arquitetura

 **Robustez**: Falhas em treinamento não derrubam API

### Negativas

 **Complexidade**: Code mais complexo com async/await

 **Sincronização**: State management requer cuidado

 **Debugging**: Mais difícil debugar code assíncrono

---

## State Management

### TrainingState

```python
class TrainingState:
    status: TrainingStatus          # IDLE, PREPARING, TRAINING, COMPLETED, FAILED
    current_epoch: int              # Época atual
    total_epochs: int               # Total de épocas
    current_loss: float             # Loss do epoch
    val_loss: float | None          # Loss de validação
    error_message: str | None       # Mensagem de erro
```

### Acesso Seguro

```python
@property
def state(self) -> TrainingState:
    return self._state  # Snapshot atual
```

Leitura é segura (Python GIL protege), mas escrita é restrita ao `TrainingService`.

---

## Tratamento de Erros

```python
try:
    await asyncio.to_thread(self._train_loop, ...)
    self._state.status = TrainingStatus.COMPLETED
except Exception as exc:
    self._state.status = TrainingStatus.FAILED
    self._state.error_message = str(exc)
    raise
```

Erros são capturados e persistidos em estado.

---

## Endpoints API

### Iniciar Treinamento

```
POST /training/start
{
    "num_epochs": 100,
    "learning_rate": 0.001
}
```

### Consultar Status

```
GET /training/status
{
    "status": "training",
    "current_epoch": 45,
    "total_epochs": 100,
    "current_loss": 0.0234,
    "val_loss": 0.0256
}
```

### Parar Treinamento (futuro)

```
POST /training/stop
```

---

## Implementação

**Arquivo**: `app/service/training_service.py`

**Uso**: `app/main.py`

```python
@app.post("/training/start")
async def start_training_endpoint(request: TrainingRequest):
    await training_service.start_training(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=request.num_epochs,
        learning_rate=request.learning_rate,
    )
    return {"status": "started"}
```

---

## Referências

- Implementação: `app/service/training_service.py`
- API Integration: `app/main.py`
- Config: `app/core/config.py`
