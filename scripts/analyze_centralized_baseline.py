"""
Analisa os resultados da baseline centralizada: avaliação por classe,
correlação instâncias x mAP50, e curva de treino.

Rodar depois de scripts/train_centralized.py, sem retreinar nada.

Saídas em results/baseline_centralized/:
  - per_class_metrics.json
  - per_class_metrics.png
  - training_curve.png
  - correlation_analysis.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats
from ultralytics import YOLO

BEST_PT = Path("runs/centralized_baseline/yolov8n_taco10/weights/best.pt")
RESULTS_CSV = Path("runs/centralized_baseline/yolov8n_taco10/results.csv")
TEST_DATA_YAML = Path("taco_data/data/test_global/data.yaml")
OUTPUT_DIR = Path("results/baseline_centralized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_class_instance_counts():
    """Instâncias globais por classe, a partir do TACO-10 reagrupado."""
    with open("taco_data/data/processed/taco10/annotations_regrouped.json") as f:
        data = json.load(f)
    names = {c["id"]: c["name"] for c in data["categories"]}
    counts = {name: 0 for name in names.values()}
    for ann in data["annotations"]:
        counts[names[ann["category_id"]]] += 1
    return counts


def per_class_evaluation():
    model = YOLO(str(BEST_PT))
    metrics = model.val(data=str(TEST_DATA_YAML), split="test", plots=True)

    instance_counts = load_class_instance_counts()
    rows = []
    for i, name in metrics.names.items():
        rows.append(
            {
                "class": name,
                "instances_global": instance_counts.get(name, None),
                "map50": float(metrics.box.ap50[i]),
                "map50_95": float(metrics.box.ap[i]),
            }
        )
    rows.sort(key=lambda r: r["map50"], reverse=True)

    with open(OUTPUT_DIR / "per_class_metrics.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # Correlação instâncias x mAP50
    inst = [r["instances_global"] for r in rows]
    m50 = [r["map50"] for r in rows]
    r_pearson, p_pearson = stats.pearsonr(inst, m50)
    r_spearman, p_spearman = stats.spearmanr(inst, m50)
    corr = {
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
        "n_classes": len(rows),
    }
    with open(OUTPUT_DIR / "correlation_analysis.json", "w") as f:
        json.dump(corr, f, indent=2)

    # Gráfico de barras por classe
    fig, ax = plt.subplots(figsize=(9, 5))
    classes = [r["class"] for r in rows]
    values = [r["map50"] for r in rows]
    colors = plt.cm.RdYlGn(np.array(values) / max(values))
    ax.barh(classes, values, color=colors)
    ax.set_xlabel("mAP50")
    ax.set_title("mAP50 por classe — baseline centralizada (teste global)")
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_class_metrics.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "per_class_metrics.svg")
    plt.close()

    print(f"Pearson r={r_pearson:.3f} (p={p_pearson:.3f})")
    print(f"Spearman r={r_spearman:.3f} (p={p_spearman:.3f})")
    return rows, corr


def training_curve():
    if not RESULTS_CSV.exists():
        print(f"Aviso: {RESULTS_CSV} não encontrado, pulando curva de treino.")
        return
    import csv

    epochs, train_box, val_box, map50, map50_95 = [], [], [], [], []
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_box.append(float(row["train/box_loss"]))
            val_box.append(float(row["val/box_loss"]))
            map50.append(float(row["metrics/mAP50(B)"]))
            map50_95.append(float(row["metrics/mAP50-95(B)"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(epochs, train_box, label="train/box_loss")
    ax1.plot(epochs, val_box, label="val/box_loss")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss de bounding box")
    ax1.legend()

    ax2.plot(epochs, map50, label="mAP50")
    ax2.plot(epochs, map50_95, label="mAP50-95")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("mAP")
    ax2.set_title("mAP na validação interna")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curve.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "training_curve.svg")
    plt.close()


if __name__ == "__main__":
    print("Avaliando por classe...")
    rows, corr = per_class_evaluation()
    print("Gerando curva de treino...")
    training_curve()
    print(f"\nSaídas em {OUTPUT_DIR}/")
