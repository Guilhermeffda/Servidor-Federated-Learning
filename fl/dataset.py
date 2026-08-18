"""
fl/dataset.py
Resolve o dataset (data.yaml no formato Ultralytics) usado por um veículo simulado
para treinar/avaliar seu YOLOv8n localmente.
"""

from pathlib import Path

VEHICLE_PARTITIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "vehicles"


def resolve_vehicle_dataset(vehicle_id: str) -> str:
    """
    Retorna o caminho do data.yaml da partição de dados do veículo `vehicle_id`.

    ALINHAR COM P2: hoje espera um data.yaml por veículo em
    data/vehicles/<vehicle_id>/data.yaml (formato padrão do Ultralytics YOLO). Esse
    diretório ainda não existe no repositório -- falta combinar com P2 como as
    partições do pLitterStreet serão entregues (uma pasta por veículo? por região
    geográfica? um único data.yaml com splits nomeados?). Ajustar esta função quando
    isso estiver definido.
    """
    data_yaml = VEHICLE_PARTITIONS_DIR / vehicle_id / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Partição de dados do veículo '{vehicle_id}' não encontrada em "
            f"{data_yaml}. ALINHAR COM P2: aguardando entrega das partições do "
            "pLitterStreet por veículo."
        )
    return str(data_yaml)


def resolve_smoke_test_dataset() -> str:
    """
    Retorna o data.yaml do dataset 'coco8' (embutido no Ultralytics, baixado
    automaticamente). Usado só para testar o pipeline FL (cliente + treino local +
    troca de pesos) antes das partições reais do pLitterStreet estarem disponíveis --
    não é o dataset do projeto.
    """
    return "coco8.yaml"
