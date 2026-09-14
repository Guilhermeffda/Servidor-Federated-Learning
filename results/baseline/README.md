# Baseline FedAvg

Cada execucao gera uma pasta `<cenario>_rep<numero>` contendo:

- `config.json`: configuracao e seed resolvidas;
- `partitions/partition.json`: atribuicao auditavel das imagens aos clientes;
- `clients.csv`: loss local e quantidade usada em cada round;
- `rounds.csv`: metricas globais por round (inclui round zero);
- `convergence.png`: curvas de convergencia em formato consistente;
- `status.json`: estado, duracao, versoes e metrica final;
- `final.pt`: checkpoint agregado local (ignorado pelo Git devido ao tamanho e nao
  exigido pela validacao de artefatos versionaveis).

Na raiz, `summary.csv` lista as seis execucoes e `aggregate.csv` traz media e desvio
padrao por cenario.

## Executar

```bash
.venv/bin/python run_config.py --all
```

Ou uma execucao isolada:

```bash
.venv/bin/python run_config.py --scenario iid --repetition 1
```

Valide os seis artefatos e a consistencia das curvas com:

```bash
.venv/bin/python scripts/validate_baseline.py
```

O arquivo `configs/baseline.yaml` usa o mini-subset disponivel no checkout e marca
`scientific_valid: false`. Esses seis resultados validam o pipeline, mas nao sao a
baseline cientifica TACO-10. Depois de preparar o TACO real, execute a configuracao
final (cinco clientes, 50 rounds e cinco epocas locais) em GPU:

```bash
python run_config.py --config configs/baseline_taco.yaml --all
```
