"""Gera as figuras do artigo (dataset, augmentação, resultados qualitativos e limiar).

Executar a partir da raiz do projeto:
    python docs/artigo/figures/make_figures.py

Reutiliza app.domain.mask_utils e app.domain.unet. Usa apenas PIL/numpy/torch.
As imagens são salvas com nomes ASCII limpos em docs/artigo/figures/.
"""

from __future__ import annotations

import glob
import os
import random
import sys

import numpy as np
from PIL import Image

# Permite importar o pacote app a partir da raiz do projeto.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from app.domain.mask_utils import decode_rgb_mask  # noqa: E402

# Caminhos
TRAIN_RGB = os.path.join(ROOT, "data", "training", "rgb", "rgb")
TRAIN_LBL = os.path.join(ROOT, "data", "training", "labels", "labels")
VAL_LBL = os.path.join(ROOT, "data", "validation", "labels", "labels")
VAL_OUT = os.path.join(ROOT, "output", "validation_run14")
AUG_GRID = os.path.join(ROOT, "output", "pre-view", "anthill_duplicate_grid.png")
CKPT = os.path.join(ROOT, "best_model_params_run14.pth")
OUT = os.path.join(ROOT, "docs", "artigo", "figures")

os.makedirs(OUT, exist_ok=True)
RED = np.array([220, 30, 30], dtype=np.float32)


def overlay(rgb: np.ndarray, binmask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Sobrepõe vermelho semitransparente onde binmask==1."""
    out = rgb.astype(np.float32).copy()
    m = binmask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * RED
    return np.clip(out, 0, 255).astype(np.uint8)


def hconcat(arrs: list[np.ndarray], sep: int = 4) -> np.ndarray:
    """Concatena imagens horizontalmente com separadores brancos."""
    h = arrs[0].shape[0]
    white = np.full((h, sep, 3), 255, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, a in enumerate(arrs):
        parts.append(a)
        if i < len(arrs) - 1:
            parts.append(white)
    return np.concatenate(parts, axis=1)


def vconcat(arrs: list[np.ndarray], sep: int = 8) -> np.ndarray:
    """Concatena imagens verticalmente com separadores brancos."""
    w = arrs[0].shape[1]
    white = np.full((sep, w, 3), 255, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, a in enumerate(arrs):
        parts.append(a)
        if i < len(arrs) - 1:
            parts.append(white)
    return np.concatenate(parts, axis=0)


def gt_bin_from_label(path: str) -> np.ndarray:
    lab = decode_rgb_mask(Image.open(path))  # 0/1/255
    return (lab == 1).astype(np.uint8), lab


def save(arr: np.ndarray, name: str) -> None:
    Image.fromarray(arr).save(os.path.join(OUT, name))
    print("  salvo:", name)


# ----------------------------------------------------------------------------
# 1) DATASET: tile positivo + máscara + negativo de solo avermelhado
# ----------------------------------------------------------------------------
def make_dataset():
    print("[1] Dataset...")
    label_files = sorted(glob.glob(os.path.join(TRAIN_LBL, "*.png")))
    pos = None  # (stem, area)
    neg_red = None  # (stem, reddishness)
    for i, lf in enumerate(label_files):
        if i > 1500 and pos is not None and neg_red is not None:
            break
        stem = os.path.splitext(os.path.basename(lf))[0]
        try:
            gtb, _ = gt_bin_from_label(lf)
        except Exception:
            continue
        area = int(gtb.sum())
        if pos is None and 1000 <= area <= 4000:
            pos = (stem, area)
        if area == 0:
            rgb_path = os.path.join(TRAIN_RGB, stem + ".jpg")
            if os.path.exists(rgb_path):
                a = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
                redness = float(a[:, :, 0].mean() - a[:, :, 1].mean())
                if neg_red is None or redness > neg_red[1]:
                    neg_red = (stem, redness, rgb_path)
    if pos is None:  # fallback: maior área disponível
        best = max(label_files[:1500], key=lambda f: int(gt_bin_from_label(f)[0].sum()))
        pos = (os.path.splitext(os.path.basename(best))[0], 0)

    stem = pos[0]
    rgb_path = os.path.join(TRAIN_RGB, stem + ".jpg")
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    gtb, lab = gt_bin_from_label(os.path.join(TRAIN_LBL, stem + ".png"))
    # máscara colorizada: formigueiro=vermelho, fundo=preto, ignorar=cinza
    mask_rgb = np.zeros_like(rgb)
    mask_rgb[lab == 1] = [220, 30, 30]
    mask_rgb[lab == 255] = [128, 128, 128]
    save(rgb, "tile_rgb.png")
    save(mask_rgb, "tile_mask.png")
    print("  positivo:", stem, "area=", pos[1])
    if neg_red is not None:
        neg = np.array(Image.open(neg_red[2]).convert("RGB"))
        save(neg, "tile_neg.png")
        print("  negativo:", neg_red[0], "redness=%.1f" % neg_red[1])


# ----------------------------------------------------------------------------
# 2) AUGMENTAÇÃO: copiar a grade pronta
# ----------------------------------------------------------------------------
def _duplicate_anthill(rgb: np.ndarray, orig_bin: np.ndarray, max_copies: int = 2):
    """Duplicação intra-recorte sem scipy: copia o formigueiro (bbox) para
    novas posições, com rotação/espelhamento, evitando sobrepor o original.
    Retorna (rgb_aumentado, máscara_das_novas_cópias) ou None."""
    H, W = orig_bin.shape
    ys, xs = np.where(orig_bin > 0)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    stamp_rgb0 = rgb[y0:y1, x0:x1].copy()
    stamp_m0 = orig_bin[y0:y1, x0:x1].copy()
    aug = rgb.copy()
    newmask = np.zeros((H, W), np.uint8)
    occupied = orig_bin.astype(bool).copy()
    placed = 0
    target = random.randint(1, max_copies)
    for _ in range(60):
        if placed >= target:
            break
        k = random.randint(0, 3)
        s_rgb = np.rot90(stamp_rgb0, k)
        s_m = np.rot90(stamp_m0, k)
        if random.random() < 0.5:
            s_rgb, s_m = np.fliplr(s_rgb), np.fliplr(s_m)
        ph, pw = s_m.shape
        if ph >= H or pw >= W:
            continue
        yy, xx = random.randint(0, H - ph), random.randint(0, W - pw)
        if occupied[yy:yy + ph, xx:xx + pw].any():
            continue  # evita sobreposição
        sm = s_m.astype(bool)
        aug[yy:yy + ph, xx:xx + pw][sm] = s_rgb[sm]
        newmask[yy:yy + ph, xx:xx + pw][sm] = 1
        occupied[yy:yy + ph, xx:xx + pw][sm] = True
        placed += 1
    return (aug, newmask) if placed > 0 else None


def make_aug():
    print("[2] Augmentação (4 exemplos, 2x2)...")
    from app.domain.mask_utils import get_anthill_binary_mask

    label_files = sorted(glob.glob(os.path.join(TRAIN_LBL, "*.png")))
    cells: list[np.ndarray] = []
    seen_scenes: set[str] = set()
    for lf in label_files:
        if len(cells) >= 4:
            break
        stem = os.path.splitext(os.path.basename(lf))[0]
        scene = "_".join(stem.split("_")[:-2])  # remove coords do tile
        if scene in seen_scenes:  # 1 exemplo por cena (diversidade)
            continue
        rgb_path = os.path.join(TRAIN_RGB, stem + ".jpg")
        if not os.path.exists(rgb_path):
            continue
        try:
            mask_img = Image.open(lf).convert("RGB")
            orig_bin = get_anthill_binary_mask(np.array(mask_img))
        except Exception:
            continue
        if not (300 <= int(orig_bin.sum()) <= 4000):
            continue
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        made = _duplicate_anthill(rgb, orig_bin, max_copies=2)
        if made is None:
            continue
        cell = hconcat([rgb, overlay(made[0], made[1], alpha=0.45)])
        cells.append(cell)
        seen_scenes.add(scene)
        print("  exemplo:", stem)

    if not cells:
        print("  AVISO: nenhum exemplo de duplicação gerado")
        return
    while len(cells) < 4:  # completa o 2x2 com branco se faltar
        cells.append(np.full_like(cells[0], 255))
    gap = np.full((cells[0].shape[0], 8, 3), 255, dtype=np.uint8)
    row1 = np.concatenate([cells[0], gap, cells[1]], axis=1)
    row2 = np.concatenate([cells[2], gap, cells[3]], axis=1)
    save(vconcat([row1, row2], sep=8), "aug_anthill_duplicate.png")


# ----------------------------------------------------------------------------
# 3) QUALITATIVO: acerto / falso negativo / falso positivo
# ----------------------------------------------------------------------------
def _pred_bin(stem: str) -> np.ndarray | None:
    p = os.path.join(VAL_OUT, stem + "_mask.png")
    if not os.path.exists(p):
        return None
    arr = np.array(Image.open(p).convert("L"))
    return (arr > 127).astype(np.uint8)


def _val_rgb(stem: str) -> np.ndarray | None:
    p = os.path.join(VAL_OUT, stem + "_rgb.png")
    if not os.path.exists(p):
        return None
    return np.array(Image.open(p).convert("RGB"))


def make_qualitative():
    print("[3] Resultados qualitativos...")
    gt_files = sorted(glob.glob(os.path.join(VAL_LBL, "*.png")))
    best_succ = None  # (iou, stem)
    best_fn = None     # (gt_area, stem)
    best_fp = None     # (pred_area, stem)
    for gf in gt_files:
        stem = os.path.splitext(os.path.basename(gf))[0]
        pred = _pred_bin(stem)
        if pred is None:
            continue
        try:
            gtb, lab = gt_bin_from_label(gf)
        except Exception:
            continue
        valid = lab != 255
        gtb = gtb & valid
        pred = pred & valid
        gt_area = int(gtb.sum())
        pred_area = int(pred.sum())
        inter = int((gtb & pred).sum())
        union = int((gtb | pred).sum())
        iou = inter / union if union > 0 else 0.0
        if gt_area > 200 and iou > 0.35:
            if best_succ is None or iou > best_succ[0]:
                best_succ = (iou, stem)
        if gt_area > 400 and pred_area == 0:
            if best_fn is None or gt_area > best_fn[0]:
                best_fn = (gt_area, stem)
        if gt_area == 0 and pred_area > 200:
            if best_fp is None or pred_area > best_fp[0]:
                best_fp = (pred_area, stem)

    def build(stem: str, name: str):
        rgb = _val_rgb(stem)
        gtb, _ = gt_bin_from_label(os.path.join(VAL_LBL, stem + ".png"))
        pred = _pred_bin(stem)
        if rgb is None or pred is None:
            print("  AVISO: faltam arquivos para", stem)
            return
        panel = hconcat([rgb, overlay(rgb, gtb), overlay(rgb, pred)])
        save(panel, name)

    for tag, sel, name in [
        ("acerto", best_succ, "qual_success.png"),
        ("falso negativo", best_fn, "qual_fn.png"),
        ("falso positivo", best_fp, "qual_fp.png"),
    ]:
        if sel is None:
            print("  AVISO: nenhum exemplo de", tag, "encontrado")
            continue
        print("  %s: %s (%s)" % (tag, sel[1], sel[0]))
        build(sel[1], name)


# ----------------------------------------------------------------------------
# 4) LIMIAR: mesma imagem segmentada a 0,50 / 0,40 / 0,35
# ----------------------------------------------------------------------------
def make_threshold():
    print("[4] Limiar (inferência Run 14)...")
    try:
        import torch
        from app.domain.unet import UNet
    except Exception as e:
        print("  AVISO: torch/UNet indisponível:", e)
        return
    if not os.path.exists(CKPT):
        print("  AVISO: checkpoint não encontrado:", CKPT)
        return

    # escolhe um tile positivo com área razoável
    gt_files = sorted(glob.glob(os.path.join(VAL_LBL, "*.png")))
    chosen = None
    for gf in gt_files:
        stem = os.path.splitext(os.path.basename(gf))[0]
        if not os.path.exists(os.path.join(VAL_OUT, stem + "_rgb.png")):
            continue
        try:
            gtb, _ = gt_bin_from_label(gf)
        except Exception:
            continue
        if 800 <= int(gtb.sum()) <= 6000:
            chosen = stem
            break
    if chosen is None:
        print("  AVISO: nenhum tile positivo adequado")
        return
    print("  tile:", chosen)

    rgb = np.array(Image.open(os.path.join(VAL_OUT, chosen + "_rgb.png")).convert("RGB"))
    net = UNet(n_channels=3, n_classes=2)
    state = torch.load(CKPT, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    try:
        net.load_state_dict(state)
    except Exception:
        net.load_state_dict(state, strict=False)
        print("  (carregado com strict=False)")
    net.eval()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (rgb.astype(np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        logits = net(x)
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

    for thr, name in [(0.50, "thr_050.png"), (0.40, "thr_040.png"), (0.35, "thr_035.png")]:
        binm = (prob >= thr).astype(np.uint8)
        save(overlay(rgb, binm), name)
        print("  limiar %.2f: %d px positivos" % (thr, int(binm.sum())))


if __name__ == "__main__":
    steps = sys.argv[1:] or ["dataset", "aug", "qual", "threshold"]
    if "dataset" in steps:
        make_dataset()
    if "aug" in steps:
        make_aug()
    if "qual" in steps:
        make_qualitative()
    if "threshold" in steps:
        make_threshold()
    print("Concluído. Figuras em docs/artigo/figures/")
