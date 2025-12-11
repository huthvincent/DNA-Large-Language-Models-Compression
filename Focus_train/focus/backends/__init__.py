"""Backend-specific adapters for focus-token integration."""

try:  # pragma: no cover - package import
    from .hf_adapter import HFAdapter
except ImportError:  # pragma: no cover - script execution
    from hf_adapter import HFAdapter

try:  # pragma: no cover - optional dependency
    try:
        from .bionemo_adapter import BioNeMoAdapter
    except ImportError:
        from bionemo_adapter import BioNeMoAdapter
except NotImplementedError:  # pragma: no cover - placeholder backend
    BioNeMoAdapter = None

__all__ = ["HFAdapter", "BioNeMoAdapter"]
