"""
Entrypoints retirados del flujo activo.
"""

from __future__ import annotations

from monai_pipeline.core.config import DIR_METRICAS
from monai_pipeline.core.utils import guardar_json


def _emitir_fase_retirada(nombre_fase: str, ruta_salida: str, motivo: str) -> None:
    guardar_json(
        {
            "estado": "retirada",
            "fase": nombre_fase,
            "motivo": motivo,
        },
        DIR_METRICAS / ruta_salida,
    )
    print(f"{nombre_fase} retirada. Ver {ruta_salida}.")


def run_reconstruction_phase() -> None:
    _emitir_fase_retirada(
        "Fase 2",
        "f2_deprecada.json",
        (
            "El proyecto ya no reconstruye CT a partir de sinogramas. "
            "El pipeline activo se centra en segmentacion tumoral axial 2D."
        ),
    )


def run_registration_phase() -> None:
    _emitir_fase_retirada(
        "Fase 3",
        "f3_deprecada.json",
        (
            "El registro interpaciente se elimina del flujo principal para evitar "
            "coste computacional y complejidad que no mejoran la segmentacion."
        ),
    )


def run_radiomics_phase() -> None:
    _emitir_fase_retirada(
        "Fase 5",
        "f5_deprecada.json",
        (
            "La radiomica queda fuera del pipeline operativo hasta tener una "
            "segmentacion suficientemente robusta y una tarea clinica bien definida."
        ),
    )
