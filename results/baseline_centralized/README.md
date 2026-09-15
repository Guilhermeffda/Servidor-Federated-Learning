# Baseline centralizada TACO-10

O pipeline foi validado com o split oficial `960/240/300`, mas o treino final de 50
epocas deve ser executado no Colab/GPU conforme a decisao do benchmark real. O arquivo
`metrics.json` so sera criado depois que as 50 epocas e a avaliacao das 300 imagens do
teste global terminarem com sucesso; nao ha metricas sinteticas ou parciais aqui.

No Colab:

```bash
python scripts/train_centralized.py --epochs 50 --imgsz 640 --batch 16 --device 0
```

Saidas esperadas: `metrics.json`, `training_status.json`, `training_curve.csv` e
`best.pt` (o checkpoint e ignorado pelo Git).
