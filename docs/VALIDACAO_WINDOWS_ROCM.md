# Validação Windows nativo / RX 7600

Validação realizada em 27–28/09/2026. **Os oito gates passaram na RX 7600.**
Os logs desta máquina estão em `F:\sfl\`; execução e dados em
`F:\sfl\project`; ambiente em `F:\sfl\.venv`. O checkout editado é o
repositório original no OneDrive, branch `fix/windows-amd-rocm`.

## Causa e solução

Repro sem workaround: Conv2d passou em `cuda:0`; BatchNorm2d falhou em
`MIOpenBatchNormFwdTrainSpatial.cpp`, `HIPRTC_ERROR_COMPILATION`, header C++
`type_traits` ausente, terminando em `miopenStatusUnknownError`.
Log: `F:\sfl\batchnorm_raw.log`. Esse diagnóstico coincide com
[ROCm issue 6150](https://github.com/ROCm/legacy-rocm-build/issues/6150).

O [fallback documentado MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/docs-7.2.2/how-to/use-nhwc-batchnorm-in-pytorch.html)
usando `torch.backends.cudnn.flags(enabled=False)` só no forward BatchNorm
passou: backward de BatchNorm e Conv2d na GPU, gradientes presentes e finitos.
Log: `F:\sfl\batchnorm_native.log`.

Passar `False` como último argumento de `torch.batch_norm` **não** evitou MIOpen
nesse build. Foi testado separadamente, sem inferir funcionamento da assinatura.
A operação ATen `torch.native_batch_norm` passou forward/backward na GPU mantendo
`torch.backends.cudnn.enabled=True` (`batchnorm_native_op.log`).

`fl/runtime.py` centraliza seleção e escopo: Windows + HIP não nulo + GPU
disponível, solicitação GPU e input em cuda. Um wrapper temporário em
`torch.batch_norm` chama ATen nativo; a validação de `F.batch_norm`, parâmetros,
running mean/variance, arquitetura e autograd permanecem PyTorch. O lock protege
instalação/refcount, ContextVar limita a seleção à thread/contexto ativo, e
`finally` restaura o callable ao terminar o último contexto. Não modifica flags
globais, Conv2d ou sincronização dos pesos.

Um segundo problema apareceu no cliente real 640/batch16: o treino terminou sem
alterar parâmetros. A instrumentação (`gradient_probe.log`) mostrou três passos
descartados por AMP: escalas 65536 → 32768 → 16384 → 8192, com 164/163/142
parâmetros com gradientes não finitos. Os buffers de BatchNorm mudaram, mas o
termo de distância dos parâmetros era zero; sincronização estava correta.

Foi validado init_scale=1024 (`amp1024_probe.log`): os três passos tiveram zero
gradientes não finitos e atualizaram pesos; 183 parâmetros treináveis mudaram,
distância proximal 2.873362. `configure_amp_scaler` usa state_dict/load_state_dict
do GradScaler para limitar a escala inicial somente em novo treino Windows ROCm
GPU, sem alterar AMP, batch, nbs, épocas, LR, loss ou acumulação. Resume preserva
a escala do checkpoint, e demais hardwares/CPU/scaler desabilitado são no-op.
Loss scaling é desfeito antes do optimizer.step; os mecanismos de backoff/growth
continuam iguais. Consulte [AMP PyTorch](https://docs.pytorch.org/docs/main/notes/amp_examples.html).

Escopo suportado: YOLO eager, cliente e avaliação, benchmark, centralizado e
análise. Não promete torch.compile/JIT nem SyncBatchNorm distribuído.
No novo Ultralytics também foi encontrada redução automática de batch por OOM:
o wrapper Windows ROCm esgota o retry budget antes de cada batch, de modo que o
erro original interrompe a execução. Nenhum hiperparâmetro científico foi alterado.

Na integração FedProx, o Ultralytics instalado retorna um vetor de três losses,
e o trainer soma esse vetor antes do backward. A implementação anterior somava
o escalar proximal a cada componente, aplicando efetivamente 3×mu. A correção
explícita soma os componentes antes de adicionar `mu * proximal_penalty`, uma
única vez. Isso preserva o objetivo definido pelo contrato, a referência global
congelada e o backward. Essa correção de contrato é independente do hardware:
loss escalar mantém o mesmo comportamento; loss vetorial deixa de multiplicar
o termo proximal. Portanto, as mudanças de backend só afetam Windows ROCm,
mas esta correção FedProx deliberadamente também vale para NVIDIA/CPU/MPS.
Não se deve comparar uma execução antiga afetada pelo bug com a versão corrigida
como se o objetivo efetivo fosse idêntico. Testes escalar/vetorial comprovam o
coeficiente exato; cliente real com mu=1 também atualizou pesos na GPU.

## Versões efetivas

| Componente | Versão |
|---|---|
| Windows | 11, build 26200 |
| Driver RX 7600 | 32.0.31041.1004 |
| Python | 3.12.10 x64 |
| PyTorch | 2.13.0+rocm10.0.0 |
| torchvision | 0.28.0+rocm10.0.0 |
| torchaudio (não usado no treino) | 2.11.0.2+rocm10.0.0 |
| ROCm SDK | 10.0.0, somente gfx1102 |
| HIP | 7.15.26333 |
| Ultralytics | 8.4.164 |
| Flower | 1.38.0 |
| NumPy | 2.5.2 |
| pandas | 2.3.3 |

Instalação conforme [AMD oficial](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
`pip install -r requirements.txt` com constraints preservou torch/torchvision
ROCm. Não houve instalação de CUDA wheels, DirectML, WSL ou Build Tools.
Build Tools já existiam no F: antes desta sessão; não foram usados como correção.

## Gates

| Gate | Evidência atual |
|---|---|
| 1 Python | 3.12.10 |
| 2 ROCm | RX 7600, gfx1102, 8176 MiB; HIP não nulo, cuda disponível |
| 3 Sintético | ACCELERATOR CHECK: OK; loss 0.3539395; 12.68 s; pico alocado 972800 bytes |
| 4 pytest | 35 passed, 3 warnings, 26.64 s |
| 5 Cliente TACO GPU | 192 imagens, 183 parâmetros alterados, roundtrip exato de 355 tensores; mu=0 e mu=1 passaram |
| 6 Smoke 5 clientes / 2 rounds GPU | complete, 400.898 s, 3 linhas rounds.csv e 10 clients.csv; todas as distâncias proximais > 0 |
| 7 Formato dos artefatos novos | FORMATO VALIDADO; asserts adicionais de finitude e contagem passaram |
| 8 Benchmark 640 / batch16 / 1 época GPU | 49.6667 s, pesos alterados; 1955933696 bytes alocados e 2464153600 reservados |

Smoke: mAP50 round 0 = 0.0059109110; round 1 = 0.0189411278;
round 2 = 0.0345155064. `scientific_valid=false` permanece correto para smoke,
que valida execução e não substitui os experimentos científicos completos.
Saídas em `F:\sfl\project\results\taco_smoke\iid_rep1`: config.json,
status.json, hardware.json, rounds.csv, clients.csv, convergence.png e final.pt.
O benchmark usou 1.82 GiB alocados / 2.29 GiB reservados de 7.98 GiB disponíveis;
seus 49.67 s incluem inicialização, treino e validação. A extrapolação original
do script deu 30.18 dias e decisão `colab`; é estimativa, não medição de 50 épocas.
Não foi feito teste contínuo de várias horas nem treino centralizado de 50 épocas.

Os testes incluem seis casos **reais na RX 7600** de BatchNorm1d/2d/3d,
FP32/FP16, treino/backward/avaliação, outputs, gradientes e running statistics
comparados ao fallback oficial. Casos simulados provam seleção NVIDIA, CPU, MPS,
Linux ROCm, exceções, contexto aninhado, outra thread e thread concorrente com
contexto próprio. Os testes existentes de peso e treino local em CPU passaram.
Regressão FedProx confirma gradiente proximal e referência congelada.

**NVIDIA preservado por design + testes unitários, não validado fisicamente
nesta máquina.** CPU foi testado fisicamente. MPS não foi testado fisicamente.

Warnings dos testes: duas deprecações Typer/Click e um aviso de que
`cuDNN benchmark_limit` não se aplica a MIOpen no teste de referência; não
impediram os checks. Não são erros HIPRTC nem OOM.

No smoke, MIOpen registrou uma mensagem `Invalid elapsed time` ao avaliar um
solver de convolução. Foi recuperada pelo autotuner, sem exceção Python,
interrupção, NaN de métricas ou perda de updates. O código oficial captura a
exceção por candidato e continua a seleção:
[solver_finders.cpp](https://github.com/ROCm/rocm-libraries/blob/develop/projects/miopen/src/conv/solver_finders.cpp).
Não foi suprimida e pode afetar desempenho. Não houve a falha fatal HIPRTC
BatchNorm na execução protegida. Outros avisos: xnackOff não suportado no
compilador gfx1102; acesso a imagens ocasionalmente lento no disco F:;
classes raras para estratificação sklearn (dados originais preservados);
optimizer=auto determina LR/momentum conforme os defaults existentes.

## Arquivos e reprodução

- `fl/runtime.py`: seleção, BatchNorm nativo e wrapper de chamada Ultralytics.
- `fl/model.py`: envolve train/val; corrige soma vetorial FedProx; mantém `_sync_trained_weights`.
- `scripts/check_accelerator.py`: preflight real, `--raw` para regressão, JSON opcional.
- `scripts/test_taco_client.py`: device/mu explícitos e assert de pesos treináveis alterados.
- `scripts/sanity_check.py`: compatibilidade, synchronize, hardware e picos de VRAM.
- `scripts/train_centralized.py`, `scripts/analyze_centralized_baseline.py`: mesmas chamadas protegidas.
- `tests/test_runtime.py`: seleção, restauração, concorrência, FedProx, precisão e OOM.
- `constraints-windows-rocm.txt`: preserva builds AMD e fixa Ultralytics/Flower efetivamente validados, após os oito gates.
- `requirements.txt`: instrução AMD; convertido de UTF-16 para UTF-8, requisitos inalterados.
- `docs/INSTRUCOES.md`: instalação completa AMD separada de NVIDIA, gates e política de OOM.
- `.gitignore`: caches de pytest e configuração local Ultralytics.
- `run_config.py` e `fl/CONTRACT.md`: hardware.json opcional, sem alterar status.json.

Os comandos de instalação desde zero, preparação TACO e validação estão em
`docs/INSTRUCOES.md`, seções 1–5. Primeiro instale Python 3.12 x64 no F: e obtenha
um checkout curto. Não repita download se arquivo validado já existir.

Para reproduzir o checkout testado nesta máquina:

```powershell
$env:TEMP='F:\rocm-temp'
$env:TMP='F:\rocm-temp'
$env:PIP_CACHE_DIR='F:\pip-cache'
$env:PYTHONUTF8='1'
$env:YOLO_CONFIG_DIR='F:\sfl\yolo-config'
Set-Location F:\sfl\project
& F:\sfl\.venv\Scripts\Activate.ps1
python scripts/check_accelerator.py --output results/accelerator_check.json
python -m pytest
python scripts/test_taco_client.py --device 0
python run_config.py --config configs/taco_smoke.yaml --scenario iid --repetition 1 --force
python scripts/validate_run_format.py --candidate results/taco_smoke/iid_rep1
python scripts/sanity_check.py --epochs 1 --imgsz 640 --batch 16 --device 0
python scripts/test_taco_client.py --device 0 --mu 1
```

Para instalar Python desde zero (instalador oficial), antes da seção AMD do guia:

```powershell
New-Item -ItemType Directory -Force F:\rocm-temp,F:\sfl | Out-Null
Invoke-WebRequest https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe -OutFile F:\rocm-temp\python-3.12.10-amd64.exe
Start-Process -FilePath F:\rocm-temp\python-3.12.10-amd64.exe -ArgumentList '/quiet InstallAllUsers=0 TargetDir=F:\Python312 Include_launcher=0 Include_test=0 PrependPath=0' -WindowStyle Hidden -Wait
git clone https://github.com/Guilhermeffda/Servidor-Federated-Learning.git F:\sfl\project
```

O clone remoto sozinho não contém estas mudanças locais: aplique o diff desta
branch ou use este checkout antes de executar os comandos AMD do guia.
Depois da instalação, execute os quatro comandos de preparação TACO da seção 2
do guia, na ordem, e então os gates acima. Ambiente efetivo completo preservado
em `docs/evidence/windows-rocm/environment-freeze.txt`. Logs e JSONs pequenos
também foram copiados para essa pasta; dataset, venv e checkpoints ficam no F:.
Nas cópias dos logs, apenas escapes ANSI de cores/progresso e espaços finais
foram removidos para leitura e verificação Git. Os logs brutos permanecem no F:.

Dados efetivos: archive TACO validado com MD5/SHA256, 1500 imagens e 4784
anotações, mapping oficial 60→10 classes, 960/240/300 no centralizado;
192/48 por cliente IID. As duas IDs de anotação duplicadas originais e as
classes escassas não foram filtradas. Configs, split seeds, LR, batch, nbs,
épocas e arquitetura não foram editados.

## Git

Branch local `fix/windows-amd-rocm`, base `976fb3f`. Nenhum push realizado.
Os commits locais solicitados estão pendentes: Git não tem user.name/user.email
configurados. Foi solicitado ao usuário o nome/e-mail; nenhuma identidade foi
inventada. Mudanças preparadas no staging. Os snapshots `git-status.txt` e
`git-diff-stat.txt` registram o estado final de implementação (antes de eventual
configuração de autor). `git diff HEAD --check` passou.
