"""Registro basado en decoradores para intercambiar modelos/pérdidas a través de Hydra.

Uso (B3+):

    @register("model", "segresnet")
    def build_segresnet(cfg) -> torch.nn.Module: ...
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=Callable[..., object])

_REGISTRY: dict[str, dict[str, Callable[..., object]]] = {}


def register(kind: str, name: str) -> Callable[[T], T]:
    bucket = _REGISTRY.setdefault(kind, {})

    def _decorator(fn: T) -> T:
        if name in bucket:
            raise KeyError(f"{kind!r}/{name!r} already registered")
        bucket[name] = fn
        return fn

    return _decorator


def resolve(kind: str, name: str) -> Callable[..., object]:
    try:
        return _REGISTRY[kind][name]
    except KeyError as exc:
        raise KeyError(f"unknown {kind}: {name!r}") from exc
