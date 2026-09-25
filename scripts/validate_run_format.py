"""
Valida que uma execução de run_config.py produz artefatos no mesmo formato
esperado do contrato do projeto (fl/CONTRACT.md) — ou, se fornecida, no mesmo
formato exato de uma execução de referência (ex.: uma execução de P4).

Fecha o DoD do card de P5: "Teste de execução de 1 configuração, formato idêntico
ao de P4 (comparação direta dos arquivos gerados)."

Uso:
    # Contra uma execução de referência já existente
    python scripts/validate_run_format.py \
        --reference results/baseline_taco/iid_rep1 \
        --candidate results/taco_smoke/iid_rep1

    # Sem referência disponível ainda: valida contra o contrato documentado
    python scripts/validate_run_format.py --candidate results/taco_smoke/iid_rep1
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Contrato de fallback, derivado diretamente de run_config.py e fl/model.py (não
# apenas do fl/CONTRACT.md em prosa, que já mostrou divergir do código real).
# Revalidado em 22/09/2026 contra a versão atual do repositório. Se o código mudar,
# reveja este dicionário antes de confiar no resultado do script.
EXPECTED_SCHEMA = {
    "config.json": {
        "type": "json",
        # Igual ao YAML de configuração (dataset_yaml, dataset_label, model, ...)
        # mais scenario/repetition/seed, adicionados por run_one(). A chave do
        # número de rounds é "rounds", não "num_rounds".
        "required_keys": {
            "dataset_yaml",
            "dataset_label",
            "scientific_valid",
            "model",
            "num_clients",
            "rounds",
            "local_epochs",
            "batch_size",
            "image_size",
            "device",
            "scenario",
            "repetition",
            "seed",
        },
    },
    "rounds.csv": {
        "type": "csv",
        # round + o retorno de evaluate_model() (fl/model.py) + elapsed_seconds,
        # adicionado por run_config.py::run_one.
        "required_columns": {
            "round",
            "loss",
            "map50",
            "map50_95",
            "precision",
            "recall",
            "num_examples",
            "elapsed_seconds",
        },
    },
    "clients.csv": {
        "type": "csv",
        # round/client_id/num_examples (adicionados por run_config.py) + o retorno
        # de train_local() (fl/model.py). Não existe coluna "loss" aqui — é
        # "train_loss". Diferente do vocabulário de rounds.csv de propósito, ver
        # fl/CONTRACT.md secao "Metricas retornadas pelo cliente".
        "required_columns": {
            "round",
            "client_id",
            "num_examples",
            "train_loss",
            "fitness",
            "mu",
            "proximal_distance_sq_half",
        },
    },
    "status.json": {
        "type": "json",
        "required_keys": {
            "run_id",
            "status",
            "scenario",
            "repetition",
            "dataset_label",
            "scientific_valid",
            "rounds_completed",
            "duration_seconds",
            "final_map50",
        },
    },
}

REQUIRED_FILES = list(EXPECTED_SCHEMA.keys())


def load_json_keys(path: Path) -> set:
    with open(path) as f:
        data = json.load(f)
    return set(data.keys())


def load_csv_columns(path: Path) -> set:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return set(h.strip() for h in header)


def check_against_schema(candidate_dir: Path) -> bool:
    ok = True
    for filename, spec in EXPECTED_SCHEMA.items():
        fpath = candidate_dir / filename
        if not fpath.is_file():
            print(f"❌ {filename}: ausente em {candidate_dir}")
            ok = False
            continue

        if spec["type"] == "json":
            keys = load_json_keys(fpath)
            missing = spec["required_keys"] - keys
            if missing:
                print(f"❌ {filename}: faltam chaves {sorted(missing)}")
                ok = False
            else:
                print(f"✅ {filename}: chaves batem")
        elif spec["type"] == "csv":
            cols = load_csv_columns(fpath)
            missing = spec["required_columns"] - cols
            if missing:
                print(f"❌ {filename}: faltam colunas {sorted(missing)}")
                ok = False
            else:
                print(f"✅ {filename}: colunas batem ({len(cols)} colunas)")
    return ok


def check_against_reference(candidate_dir: Path, reference_dir: Path) -> bool:
    ok = True
    for filename in REQUIRED_FILES:
        cpath, rpath = candidate_dir / filename, reference_dir / filename
        if not rpath.is_file():
            print(f"⚠️  {filename}: não existe na referência, pulando comparação direta")
            continue
        if not cpath.is_file():
            print(f"❌ {filename}: ausente em {candidate_dir}")
            ok = False
            continue

        spec = EXPECTED_SCHEMA[filename]
        if spec["type"] == "json":
            ck, rk = load_json_keys(cpath), load_json_keys(rpath)
            if ck != rk:
                print(f"❌ {filename}: chaves diferem — só na referência: {sorted(rk - ck)}, só no candidato: {sorted(ck - rk)}")
                ok = False
            else:
                print(f"✅ {filename}: chaves idênticas à referência")
        elif spec["type"] == "csv":
            cc, rc = load_csv_columns(cpath), load_csv_columns(rpath)
            if cc != rc:
                print(f"❌ {filename}: colunas diferem — só na referência: {sorted(rc - cc)}, só no candidato: {sorted(cc - rc)}")
                ok = False
            else:
                print(f"✅ {filename}: colunas idênticas à referência ({len(cc)} colunas)")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", required=True, type=Path,
        help="Pasta da execução a validar, ex.: results/taco_smoke/iid_rep1",
    )
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="Pasta de uma execução de referência (ex.: de P4). Se omitido, "
             "valida contra o contrato documentado em fl/CONTRACT.md.",
    )
    args = parser.parse_args()

    if not args.candidate.is_dir():
        print(f"❌ Pasta candidata não existe: {args.candidate}")
        sys.exit(1)

    if args.reference:
        if not args.reference.is_dir():
            print(f"❌ Pasta de referência não existe: {args.reference}")
            sys.exit(1)
        print(f"Comparando {args.candidate} contra referência {args.reference}\n")
        schema_ok = check_against_schema(args.candidate)
        print()
        ref_ok = check_against_reference(args.candidate, args.reference)
        ok = schema_ok and ref_ok
    else:
        print(f"Nenhuma referência fornecida — validando {args.candidate} contra o contrato documentado (fl/CONTRACT.md)\n")
        ok = check_against_schema(args.candidate)

    print()
    if ok:
        print("✅ FORMATO VALIDADO")
        sys.exit(0)
    else:
        print("❌ FORMATO INVÁLIDO — corrija antes de considerar o card de P5 concluído")
        sys.exit(1)


if __name__ == "__main__":
    main()
