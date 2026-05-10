"""Classes de métricas para validação e avaliação do modelo."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.domain.mask_utils import (
    LABEL_ANTHILL,
    LABEL_BACKGROUND,
    LABEL_IGNORE,
)


@dataclass
class ValidationMetrics:
    """Métricas agregadas da validação."""

    # Total de imagens processadas.
    total_images: int = 0

    # Quantidade de imagens com detecção.
    anthill_detections: int = 0

    # Acurácia global por pixel.
    pixel_accuracy: float = 0.0

    # Média de IoU.
    mean_iou: float = 0.0

    # Média de Dice Score.
    mean_dice: float = 0.0

    # IoU individual por imagem.
    per_image_iou: list[float] = field(default_factory=list)

    # Dice individual por imagem.
    per_image_dice: list[float] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Acumula métricas de detecção e segmentação."""

    # Total de imagens avaliadas.
    total_images: int = 0

    # Estatísticas de detecção por imagem.
    gt_positive: int = 0
    pred_positive: int = 0

    # Verdadeiros positivos.
    tp: int = 0

    # Falsos positivos.
    fp: int = 0

    # Falsos negativos.
    fn: int = 0

    # Verdadeiros negativos.
    tn: int = 0

    # Métricas por pixel.
    px_inter: list[int] = field(default_factory=lambda: [0, 0])
    px_union: list[int] = field(default_factory=lambda: [0, 0])
    px_pred_sum: list[int] = field(default_factory=lambda: [0, 0])
    px_gt_sum: list[int] = field(default_factory=lambda: [0, 0])

    # Pixels corretos e totais.
    px_correct: int = 0
    px_total: int = 0

    def update_image_level(
        self,
        gt_pos: bool,
        pred_pos: bool
    ) -> None:
        """Atualiza métricas de detecção por imagem."""

        self.total_images += 1

        if gt_pos:
            self.gt_positive += 1

        if pred_pos:
            self.pred_positive += 1

        if gt_pos and pred_pos:
            self.tp += 1

        elif not gt_pos and pred_pos:
            self.fp += 1

        elif gt_pos and not pred_pos:
            self.fn += 1

        else:
            self.tn += 1

    def update_pixel_level(
        self,
        gt: np.ndarray,
        pred: np.ndarray
    ) -> None:
        """Atualiza métricas de segmentação por pixel."""

        # Remove pixels ignorados.
        valid = gt != LABEL_IGNORE

        gt_v = gt[valid].astype(np.int32)
        pred_v = pred[valid].astype(np.int32)

        # Conta pixels corretos.
        self.px_correct += int((gt_v == pred_v).sum())

        # Conta pixels válidos.
        self.px_total += int(valid.sum())

        # Calcula métricas por classe.
        for cls in (LABEL_BACKGROUND, LABEL_ANTHILL):

            gt_cls = gt_v == cls
            pred_cls = pred_v == cls

            # Interseção.
            self.px_inter[cls] += int(
                (gt_cls & pred_cls).sum()
            )

            # União.
            self.px_union[cls] += int(
                (gt_cls | pred_cls).sum()
            )

            # Pixels previstos.
            self.px_pred_sum[cls] += int(
                pred_cls.sum()
            )

            # Pixels reais.
            self.px_gt_sum[cls] += int(
                gt_cls.sum()
            )

    def report(self) -> str:
        """Gera relatório completo das métricas."""

        sep = "=" * 62

        lines: list[str] = [
            sep,
            "  ANTHILL DETECTION — EVALUATION METRICS",
            sep,
            "",
        ]

        # Estatísticas gerais.
        lines += [
            f"  Total images evaluated:        {self.total_images}",
            f"  GT positives (label has FQ):   {self.gt_positive}",
            f"  GT negatives (label empty):    {self.total_images - self.gt_positive}",
            f"  Pred positives (detected):     {self.pred_positive}",
            f"  Pred negatives (not detected): {self.total_images - self.pred_positive}",
            "",
            "-" * 62,
            "  Image-level Detection",
            "-" * 62,
            f"  TP (correct detection):    {self.tp:>6}",
            f"  FP (false alarm):          {self.fp:>6}",
            f"  FN (missed anthill):       {self.fn:>6}",
            f"  TN (correct rejection):    {self.tn:>6}",
            "",
        ]

        # Precision.
        precision = (
            self.tp / (self.tp + self.fp)
            if (self.tp + self.fp) > 0
            else 0.0
        )

        # Recall.
        recall = (
            self.tp / (self.tp + self.fn)
            if (self.tp + self.fn) > 0
            else 0.0
        )

        # F1 Score.
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        lines += [
            f"  Precision:                 {precision:.4f}  ({precision * 100:.1f}%)",
            f"  Recall / Detection Rate:   {recall:.4f}  ({recall * 100:.1f}%)",
            f"  F1 Score:                  {f1:.4f}  ({f1 * 100:.1f}%)",
            "",
            "-" * 62,
            "  Pixel-level Segmentation",
            "-" * 62,
        ]

        # Pixel Accuracy global.
        px_acc = (
            self.px_correct / self.px_total
            if self.px_total > 0
            else 0.0
        )

        lines.append(
            f"  Pixel Accuracy:                {px_acc:.4f}  ({px_acc * 100:.1f}%)"
        )

        lines.append("")

        cls_names = ["Background", "Anthill   "]

        ious: list[float] = []
        dices: list[float] = []

        # Calcula IoU e Dice por classe.
        for cls in (LABEL_BACKGROUND, LABEL_ANTHILL):

            inter = self.px_inter[cls]
            union = self.px_union[cls]

            pred_s = self.px_pred_sum[cls]
            gt_s = self.px_gt_sum[cls]

            # IoU.
            iou = inter / union if union > 0 else 1.0

            # Dice Score.
            dice_denom = pred_s + gt_s

            dice = (
                2 * inter / dice_denom
                if dice_denom > 0
                else 1.0
            )

            ious.append(iou)
            dices.append(dice)

            lines += [
                f"  [{cls_names[cls]}]  IoU:  {iou:.4f}  ({iou * 100:.1f}%)",
                f"  [{cls_names[cls]}]  Dice: {dice:.4f}  ({dice * 100:.1f}%)",
                "",
            ]

        # Médias globais.
        mean_iou = sum(ious) / len(ious)
        mean_dice = sum(dices) / len(dices)

        lines += [
            f"  Mean IoU:                      {mean_iou:.4f}  ({mean_iou * 100:.1f}%)",
            f"  Mean Dice:                     {mean_dice:.4f}  ({mean_dice * 100:.1f}%)",
            "",
            sep,
        ]

        return "\n".join(lines)

