# Benchmark de hardware — Card 1 (Sprint 1)

Medição de referência para estimar o custo dos experimentos reais (Card 3 e Sprint 2)
e decidir se migramos para o Google Colab (plano B já acordado pelo grupo).

## Máquina

- CPU: 11th Gen Intel Core i7-1165G7 @ 2.80GHz, 8 threads lógicas
- **Sem GPU** (treino 100% em CPU)
- Python 3.12.10 (ver [pendência sobre versão do Python](../README.md#pendências-e-decisões-abertas))
- torch 2.13.0+cpu, ultralytics 8.4.121, flwr 1.33.0

## Resultado do `scripts/sanity_check.py`

- 3 épocas, `imgsz=640`, `batch=8`, YOLOv8n (transfer learning a partir de `yolov8n.pt`)
- **Tempo total: 39,2s → ~13,1s/época em média**
- Pesos salvos em `runs/detect/train-*/weights/best.pt` e recarregados com sucesso
  (`YOLO(best_weights_path)` + `model.val()` rodaram sem erro)

⚠️ **Essa medição não é representativa do custo real.** O mini-subset usado tem só
**8 imagens** (`coco8`, dataset de exemplo do Ultralytics — ver pendência do dataset
real no README). O tempo medido aqui é dominado pelo overhead fixo de setup (carregar
modelo, escanear dataset, plotar labels), não pelo tempo de forward/backward por imagem.
Com o pLitterStreet real (centenas/milhares de imagens por partição), o tempo por época
deve crescer proporcionalmente ao número de imagens — **precisa ser remedido** assim que
o mini-subset real (ou o dataset centralizado do Card 3) estiver disponível.

## Implicação para Colab

Com só 3 mAP50~0.89 rodando em 39s neste hardware, não há sinal ainda de que CPU seja
inviável — mas essa conclusão só é confiável com dados reais. Decisão de migrar para
Colab fica em aberto até remedir com o pLitterStreet.
