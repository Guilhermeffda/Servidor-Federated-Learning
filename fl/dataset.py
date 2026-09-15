"""Resolucao das particoes TACO usadas pelos clientes federados."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARTITIONS_ROOT = PROJECT_ROOT / "data" / "partitions"


def resolve_client_dataset(client_id: str, scenario: str) -> str:
    """Resolve data/partitions/{iid,non_iid}/client_X/data.yaml."""
    if scenario not in {"iid", "non_iid"}:
        raise ValueError("scenario deve ser 'iid' ou 'non_iid'")
    if not client_id.startswith("client_") or not client_id[7:].isdigit():
        raise ValueError("client_id deve seguir o formato client_X")
    data_yaml = PARTITIONS_ROOT / scenario / client_id / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Particao TACO nao encontrada: {data_yaml}")
    return str(data_yaml)
