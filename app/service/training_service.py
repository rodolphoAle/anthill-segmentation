"""Async service for model training and evaluation.

CPU / GPU-bound work is offloaded via ``asyncio.to_thread`` so the
main event loop stays responsive while heavy computation runs in
the default thread-pool executor.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.exceptions import TrainingAlreadyInProgressError
from app.domain.losses import CombinedTverskyFocalLoss


# Estados possíveis de um treinamento
class TrainingStatus(str, Enum):
    """Finite-state representation of a training job."""

    # Não está treinando
    IDLE = "idle"
    # Preparando dados e configurações
    PREPARING = "preparing"
    # Treinamento em progresso
    TRAINING = "training"
    # Treinamento concluído com sucesso
    COMPLETED = "completed"
    # Treinamento falhou
    FAILED = "failed"


# Armazena informações sobre o estado atual do treinamento
class TrainingState:
    """Mutable snapshot of the current training run."""

    def __init__(self) -> None:
        # Status atual do treinamento
        self.status: TrainingStatus = TrainingStatus.IDLE
        # Época atual em progresso
        self.current_epoch: int = 0
        # Número total de épocas a treinar
        self.total_epochs: int = 0
        # Valor da loss na última época
        self.current_loss: float = 0.0
        # Valor da loss de validação
        self.val_loss: float | None = None
        # Mensagem de erro se treinamento falhar
        self.error_message: str | None = None


#  Service 

# Serviço principal para treinamento do modelo
class TrainingService:
    """Orchestrates model training, evaluation, and persistence.

    A single instance manages **one** model and tracks training
    progress through :attr:`state`.

    Args:
        model: PyTorch ``nn.Module`` to train.
    """

    def __init__(self, model: nn.Module) -> None:
        # Modelo de rede neural a treinar
        self._model = model
        # Detecta e configurar dispositivo (CPU/GPU)
        self._device = self._resolve_device()
        # Move modelo para o dispositivo
        self._model.to(self._device)
        # Estado inicial do treinamento
        self._state = TrainingState()
        # Ativa otimizações CUDA se disponível
        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    #  helpers 

    @staticmethod
    def _resolve_device() -> torch.device:
        """Determine the computing device from settings or auto-detect."""
        # Lê configuração de dispositivo
        device_setting: str = getattr(settings, "device", "auto")
        # Se configurado como automático, detecta GPU ou usa CPU
        if device_setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Caso contrário, usa dispositivo configurado
        return torch.device(device_setting)

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> nn.Module:
        return self._model

    #  synchronous loops (run inside thread pool) 

    def _train_loop(
        self,
        train_loader: DataLoader[tuple],
        val_loader: DataLoader[tuple],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        num_epochs: int,
        scaler: torch.amp.GradScaler,
    ) -> None:
        # Armazena melhor loss de validação para salvar checkpoint
        best_val_loss = float("inf")
        # AMP desabilitado: a UNet não tem BatchNorm, então ativações FP16
        # podem fazer overflow na forward pass, causando gradientes Inf que
        # levam a NaN após clip_grad_norm. Usa FP32 puro por segurança.
        use_amp = False
        # Total de batches para calcular progresso
        total_batches = len(train_loader)
        # Log informações a cada 500 batches
        log_every = 500  # log every 500 batches regardless of dataset size

        # Loop sobre todas as épocas
        for epoch in range(num_epochs):
            # Atualiza estado da época atual
            self._state.current_epoch = epoch + 1
            # Marca tempo de início da época
            epoch_start = time.monotonic()

            # ===== FASE DE TREINAMENTO =====
            # Coloca modelo em modo treino (ativa dropout, batch norm, etc)
            self._model.train()
            # Acumula loss durante a época
            epoch_loss = 0.0
            # Conta batches processados com sucesso
            batch_count = 0
            # Controla frequência de logs
            last_log_time = time.monotonic()

            # Loop sobre cada batch de treinamento
            for batch_idx, (inputs, labels, names) in enumerate(train_loader):
                # Marca quando batch foi carregado
                fetch_done = time.monotonic()
                # Move dados para dispositivo (GPU/CPU) sem bloquear
                inputs = inputs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)

                # Verifica dados corrompidos (ex: download SSL falhou)
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    logger.warning(
                        "Epoch {}/{} batch {}  skipping: NaN/Inf in inputs ({})",
                        epoch + 1, num_epochs, batch_idx + 1, names[0],
                    )
                    last_log_time = time.monotonic()
                    # Pula este batch corrompido
                    continue

                # Verifica se todos pixels são marcados como ignoráveis
                # (pode acontecer com rotações aleatórias que cortam regiões válidas)
                if (labels == 255).all():
                    logger.warning(
                        "Epoch {}/{} batch {}  skipping: all pixels are ignore_index ({})",
                        epoch + 1, num_epochs, batch_idx + 1, names[0],
                    )
                    last_log_time = time.monotonic()
                    # Pula batch inválido
                    continue

                # Limpa gradientes anteriores
                optimizer.zero_grad()
                # Executa forward pass (com autocast desabilitado por segurança)
                with torch.amp.autocast(device_type=self._device.type, enabled=use_amp):
                    outputs = self._model(inputs)

                # Verifica saída do modelo
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.warning(
                        "Epoch {}/{} batch {}  skipping: NaN/Inf in model outputs ({})",
                        epoch + 1, num_epochs, batch_idx + 1, names[0],
                    )
                    # Limpa gradientes antes de pular
                    optimizer.zero_grad()
                    last_log_time = time.monotonic()
                    continue

                # Calcula loss combinada (Focal + Tversky + Lovász)
                loss = criterion(outputs, labels)

                # Verifica se loss é válida
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        "Epoch {}/{} batch {}  skipping: NaN/Inf loss ({})",
                        epoch + 1, num_epochs, batch_idx + 1, names[0],
                    )
                    # Limpa gradientes antes de pular
                    optimizer.zero_grad()
                    last_log_time = time.monotonic()
                    continue

                # Backward pass com scaling (mesmo que AMP esteja desabilitado)
                scaler.scale(loss).backward()
                # Remove scaling antes de clip
                scaler.unscale_(optimizer)
                # Clipagem de gradientes para evitar explosão
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    max_norm=settings.grad_clip_max_norm,
                )
                # Passo do otimizador com scaling
                scaler.step(optimizer)
                # Atualiza escala do GradScaler
                scaler.update()

                # Acumula loss do batch
                epoch_loss += loss.item()
                # Incrementa contador de batches
                batch_count += 1
                # Marca tempo de conclusão do batch
                gpu_done = time.monotonic()

                # Log a cada 500 batches ou no primeiro batch
                if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                    elapsed_epoch = gpu_done - epoch_start
                    since_last = gpu_done - last_log_time
                    fetch_ms = (fetch_done - last_log_time) * 1000
                    gpu_ms = (gpu_done - fetch_done) * 1000
                    progress_pct = (batch_idx + 1) / total_batches * 100
                    sample_names = ", ".join(names[:2])
                    logger.info(
                        "Epoch {}/{} [{:.1f}%] batch {}/{} | loss: {:.4f} | "
                        "fetch: {:.0f}ms  gpu: {:.0f}ms  step: {:.1f}s | {}",
                        epoch + 1,
                        num_epochs,
                        progress_pct,
                        batch_idx + 1,
                        total_batches,
                        epoch_loss / batch_count,
                        fetch_ms,
                        gpu_ms,
                        since_last,
                        sample_names,
                    )
                    last_log_time = gpu_done

            # Calcula média da loss para a época
            avg_loss = epoch_loss / max(batch_count, 1)
            # Armazena loss no estado
            self._state.current_loss = avg_loss

            # ===== FASE DE VALIDAÇÃO =====
            val_loss = self._evaluate_loop_sync(val_loader, criterion)
            # Armazena loss de validação no estado
            self._state.val_loss = val_loss
            # Ajusta taxa de aprendizado baseado em val_loss
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # Scheduler com monitoramento de métrica
                scheduler.step(val_loss)
            else:
                # Scheduler com passo fixo
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch {}/{}  loss: {:.4f} | val_loss: {:.4f} | lr: {:.2e}",
                epoch + 1,
                num_epochs,
                avg_loss,
                val_loss,
                current_lr,
            )

            # Salva checkpoint se melhorou loss de validação
            if val_loss < best_val_loss:
                # Atualiza melhor loss encontrada
                best_val_loss = val_loss
                # Salva pesos do modelo ao melhorar
                torch.save(
                    self._model.state_dict(),
                    settings.best_model_params_path,
                )
                logger.info(
                    "New best val_loss={:.4f}  checkpoint saved to '{}'",
                    best_val_loss,
                    settings.best_model_params_path,
                )

    # Função síncrona que evalúa o modelo em dados de validação
    def _evaluate_loop_sync(
        self,
        val_loader: DataLoader[tuple],
        criterion: nn.Module,
    ) -> float:
        # Coloca modelo em modo avaliação (desativa dropout, batch norm, etc)
        self._model.eval()
        # Acumula loss da validação
        total_loss = 0.0
        # Conta batches válidos
        batch_count = 0
        # Desativa cálculo de gradientes (economia de memória)
        with torch.no_grad():
            # Loop sobre batches de validação
            for inputs, labels, _names in val_loader:
                # Move dados para dispositivo
                inputs = inputs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                # Forward pass
                outputs = self._model(inputs)
                # Calcula loss
                loss = criterion(outputs, labels)
                # Pula batch se tiver NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                # Acumula loss válida
                total_loss += loss.item()
                # Incrementa contador
                batch_count += 1
        # Se nenhum batch válido, retorna infinito
        if batch_count == 0:
            logger.warning("Validation loop: all batches produced NaN/Inf loss  val_loss reported as inf")
            return float("inf")
        # Retorna média de loss
        return total_loss / batch_count

    def _evaluate_loop(
        self,
        val_loader: DataLoader[tuple],
        criterion: nn.Module,
    ) -> float:
        return self._evaluate_loop_sync(val_loader, criterion)


    # Inicia treinamento de forma assíncrona
    async def start_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Start training the model asynchronously.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs: Number of epochs to train (default from settings).
            learning_rate: Learning rate (default from settings).
            
        Raises:
            TrainingAlreadyInProgressError: If training is already running.
        """
        # Verifica se treinamento já está em progresso
        if self._state.status == TrainingStatus.TRAINING:
            raise TrainingAlreadyInProgressError(
                "A training job is already in progress"
            )

        # Define número de épocas
        epochs = num_epochs or settings.num_epochs
        # Define taxa de aprendizado
        lr = learning_rate or settings.learning_rate

        # Reseta estado do treinamento
        self._state = TrainingState()
        # Marca como preparando
        self._state.status = TrainingStatus.PREPARING
        # Armazena total de épocas
        self._state.total_epochs = epochs

        # ===== CONFIGURAÇÃO DA LOSS COMBINADA =====
        # Pesos para cada classe (fundo e formigueiro)
        class_weights = torch.tensor(
            [settings.class_weight_background, settings.class_weight_anthill],
            device=self._device,
        )

        # Instancia loss combinada com Focal, Tversky e Lovász
        criterion: nn.Module = CombinedTverskyFocalLoss(
            # Peso para falsos positivos na Tversky Loss
            tversky_alpha=0.4,
            # Peso para falsos negativos na Tversky Loss
            tversky_beta=0.6,
            # Fração da loss total vinda de Tversky
            tversky_weight=0.5,
            # Fração da loss total vinda de Lovász
            lovasz_weight=0.3,
            # Controla foco em exemplos difíceis na Focal Loss
            focal_gamma=2.0,
            # Pesos aplicados às classes
            class_weights=class_weights,
            # Valor que marca pixels a ignorar
            ignore_index=255,
        )

        logger.info(
            "Using Combined Loss (Focal + Tversky + Lovász) "
            "(tversky_weight=0.5, lovasz_weight=0.3, focal_weight=0.2)"
        )

        # ===== CONFIGURAÇÃO DO OTIMIZADOR =====
        # Usa Adam como otimizador
        optimizer = optim.Adam(self._model.parameters(), lr=lr)

        # ===== CONFIGURAÇÃO DO SCHEDULER =====
        # ReduceLROnPlateau reduz taxa de aprendizado quando loss não melhora
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            # Modo mínimo: reduz LR quando val_loss para de diminuir
            mode="min",
            # Fator multiplicativo para redução
            factor=settings.scheduler_factor,
            # Épocas sem melhoria antes de reduzir
            patience=settings.scheduler_patience,
        )
        logger.info(
            "LR scheduler: ReduceLROnPlateau (patience={}, factor={})",
            settings.scheduler_patience,
            settings.scheduler_factor,
        )

        # Inicializa GradScaler (mesmo desabilitado por segurança)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        #  EXECUÇÃO DO TREINAMENTO 
        try:
            # Marca estado como treinando
            self._state.status = TrainingStatus.TRAINING
            logger.info(
                "Training started — {} epochs on {}",
                epochs,
                self._device,
            )
            # Executa loop de treinamento em thread separada (não bloqueia async)
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

            # Executa validação final após conclusão
            logger.info("Training done. Running final validation…")
            val_loss = await asyncio.to_thread(
                self._evaluate_loop, val_loader, criterion,
            )
            # Armazena loss final
            self._state.val_loss = val_loss
            # Marca como concluído
            self._state.status = TrainingStatus.COMPLETED
            logger.info("Final validation loss: {:.4f}", val_loss)

        # Tratamento de erros durante treinamento
        except Exception as exc:
            # Marca como falha
            self._state.status = TrainingStatus.FAILED
            # Armazena mensagem de erro
            self._state.error_message = str(exc)
            logger.error("Training failed: {}", exc)
            # Lança exceção novamente
            raise

    # Salva pesos do modelo em arquivo
    async def save_model(self, path: str | None = None) -> str:
        """Persist model weights to disk."""
        # Define caminho: usa fornecido ou padrão das configurações
        save_path = path or settings.model_save_path
        # Executa salvamento em thread separada
        await asyncio.to_thread(
            torch.save, self._model.state_dict(), save_path,
        )
        logger.info("Model saved → {}", save_path)
        return save_path

    # Carrega pesos do modelo de um arquivo
    async def load_model(self, path: str | None = None) -> None:
        """Load model weights from disk."""
        # Define caminho: usa fornecido ou padrão das configurações
        load_path = path or settings.model_save_path
        # Carrega estado do modelo em thread separada
        state_dict = await asyncio.to_thread(
            torch.load, load_path, map_location=self._device,
        )
        # Aplica pesos ao modelo
        self._model.load_state_dict(state_dict)
        # Garante que modelo está no dispositivo correto
        self._model.to(self._device)
        logger.info("Model loaded ← {}", load_path)
