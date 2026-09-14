# Status dos cards P4 — Sprint 2

Atualizado em 14/09/2026.

## Remedir benchmark com TACO real

- [x] Particao IID real preparada (240 imagens; 192 treino + 48 validacao).
- [x] Medicao em `640`, batch 16, CPU: 160,20 s/epoca.
- [x] Estimativa das 42 execucoes: 2.336,28 h / 97,35 dias sequenciais.
- [x] Decisao registrada: Colab/GPU.
- [ ] Comunicar externamente ao grupo (o repositorio nao possui conector de mensagens).

## Portar cliente, remover dropout e suportar FedProx

- [x] Cliente e modelo ativos em `fl/`.
- [x] Resolver de `data/partitions/{iid,non_iid}/client_X/data.yaml`.
- [x] Dropout removido do caminho ativo e preservado em `sprint2_preview/`.
- [x] Termo proximal aplicado antes do backward quando `mu > 0`.
- [x] Round-trip exato dos 355 tensores.
- [x] Treino isolado concluido em uma particao TACO real com `mu=0`.

## Baseline centralizada

- [x] Dataset centralizado e teste global oficial preparados e auditados.
- [x] Runner de 50 epocas, avaliacao final e registro de metricas implementados.
- [x] Dry-run validado com 960 treino / 240 validacao / 300 teste.
- [ ] Treino final de 50 epocas no Colab e geracao de `metrics.json`.

Decisao de comparabilidade: o card menciona 70/15/15, mas o protocolo ja
documentado no repositorio reserva 20% (300 imagens) como teste global comum a todas
as estrategias. Para nao avaliar o centralizado e o federado em imagens diferentes,
foi mantido esse teste oficial; os 1.200 exemplos restantes foram divididos em 960 de
treino e 240 de validacao (resultado final: 64/16/20).

## Baseline FedAvg IID/non-IID, tres repeticoes

- [x] Runner, agregacao, CSVs e curvas implementados.
- [x] Seis smoke tests locais concluidos e marcados `scientific_valid=false`.
- [x] Configuracao cientifica TACO-10 preparada para cinco clientes e 50 rounds.
- [ ] Executar as seis baselines TACO-10 no Colab e versionar as metricas/curvas.
