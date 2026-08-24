"""
scripts/sanity_check.py
Card 1 (Sprint 1, P4) — valida que o ambiente YOLOv8n/Ultralytics está funcional
nesta máquina e mede o tempo de treino por época, referência para estimar o custo
dos experimentos federados reais (e decidir se migramos para o Google Colab).

Uso:
    python scripts/sanity_check.py
"""

import time
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parent.parent / "data" / "mini_test" / "data.yaml"
EPOCHS = 3
IMAGE_SIZE = 640
BATCH_SIZE = 8


def run_sanity_check() -> None:
    """Treina o YOLOv8n por EPOCHS épocas, valida os pesos salvos e mede o tempo/época."""
    model = YOLO("yolov8n.pt")

    start = time.perf_counter()
    model.train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMAGE_SIZE, batch=BATCH_SIZE)
    elapsed_seconds = time.perf_counter() - start

    best_weights_path = model.trainer.best
    print(f"\nPesos salvos em: {best_weights_path}")

    # Confirma que os pesos salvos são carregáveis e utilizáveis (DoD do Card 1).
    reloaded_model = YOLO(best_weights_path)
    reloaded_model.val(data=str(DATA_YAML))

    print(f"\nTempo total de treino ({EPOCHS} épocas): {elapsed_seconds:.1f}s")
    print(f"Tempo médio por época: {elapsed_seconds / EPOCHS:.1f}s")


if __name__ == "__main__":
    run_sanity_check()
