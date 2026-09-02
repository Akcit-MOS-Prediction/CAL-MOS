import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

SPLITS = {
    "Train": "/workspace/datasets/TMHINTQI/sets/train.csv",
    "Val":   "/workspace/datasets/TMHINTQI/sets/val.csv",
    "Test":  "/workspace/datasets/TMHINTQI/sets/test.csv",
}

TARGET_COL   = "mos"
FILENAME_COL = "filepath"

COLORS = {
    "Train": "#4C72B0",
    "Val":   "#55A868",
    "Test":  "#C44E52",
}

# ── Carrega os dados ──────────────────────────────────────────────────────────
dfs = {}
for split, path in SPLITS.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    dfs[split] = pd.read_csv(path)
    print(f"{split}: {len(dfs[split])} amostras  |  colunas: {list(dfs[split].columns)}")

# ── Estatísticas descritivas ──────────────────────────────────────────────────
print("\n── Estatísticas descritivas (MOS) ──")
for split, df in dfs.items():
    s = df[TARGET_COL]
    print(f"\n{split} ({len(df)} amostras)")
    print(f"  Média:    {s.mean():.4f}")
    print(f"  Mediana:  {s.median():.4f}")
    print(f"  Std:      {s.std():.4f}")
    print(f"  Min/Max:  {s.min():.4f} / {s.max():.4f}")

# ── Figura ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("TMHINTQI — Distribuição dos dados", fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

BINS = 30

# Linha 1: histogramas individuais por split
for col, (split, df) in enumerate(dfs.items()):
    ax = fig.add_subplot(gs[0, col])
    ax.hist(df[TARGET_COL], bins=BINS, color=COLORS[split], edgecolor="white",
            linewidth=0.5, alpha=0.9)
    ax.axvline(df[TARGET_COL].mean(),   color="black",  linestyle="--", linewidth=1.2, label=f"Média {df[TARGET_COL].mean():.2f}")
    ax.axvline(df[TARGET_COL].median(), color="orange", linestyle=":",  linewidth=1.2, label=f"Mediana {df[TARGET_COL].median():.2f}")
    ax.set_title(f"{split}  (n={len(df):,})", fontsize=11, fontweight="bold")
    ax.set_xlabel("MOS", fontsize=10)
    ax.set_ylabel("Frequência", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

# Linha 2-left: histogramas sobrepostos
ax_overlay = fig.add_subplot(gs[1, 0:2])
for split, df in dfs.items():
    ax_overlay.hist(df[TARGET_COL], bins=BINS, color=COLORS[split],
                    alpha=0.5, edgecolor="white", linewidth=0.4, label=split)
ax_overlay.set_title("Distribuições sobrepostas", fontsize=11, fontweight="bold")
ax_overlay.set_xlabel("MOS", fontsize=10)
ax_overlay.set_ylabel("Frequência", fontsize=10)
ax_overlay.legend(fontsize=10)
ax_overlay.grid(axis="y", alpha=0.3)

# Linha 2-right: boxplots comparativos
ax_box = fig.add_subplot(gs[1, 2])
data_box  = [df[TARGET_COL].values for df in dfs.values()]
labels    = list(dfs.keys())
bp = ax_box.boxplot(data_box, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
for patch, split in zip(bp["boxes"], labels):
    patch.set_facecolor(COLORS[split])
    patch.set_alpha(0.8)
ax_box.set_title("Boxplot por split", fontsize=11, fontweight="bold")
ax_box.set_ylabel("MOS", fontsize=10)
ax_box.grid(axis="y", alpha=0.3)

# Contagens no título dos boxplots
for i, (split, df) in enumerate(dfs.items(), start=1):
    ax_box.text(i, ax_box.get_ylim()[0] - 0.03 * (ax_box.get_ylim()[1] - ax_box.get_ylim()[0]),
                f"n={len(df):,}", ha="center", va="top", fontsize=8, color="gray")

plt.savefig("tmhintqi_distribution.png", dpi=150, bbox_inches="tight")
print("\n✅ Figura salva em: tmhintqi_distribution.png")
plt.show()