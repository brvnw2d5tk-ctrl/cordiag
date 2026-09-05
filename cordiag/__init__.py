"""Core algorithms for paired molecular-data diagnostics.

Available modules provide stratified ridge-based prediction, paired-gain
statistics, transportability-gap statistics, calibration simulations, and a
command-line interface. Each module is self-contained and can be imported
independently.
"""

from . import m1

__version__ = "0.1.3"

__all__ = ["m1", "zpg", "tg", "calibration", "cli"]

# PEP 562 lazy loading keeps the lightweight top-level import independent of
# optional module dependencies.
_LAZY_MODULES = ("zpg", "tg", "calibration", "cli")


def __getattr__(name):
    if name in _LAZY_MODULES:
        try:
            module = __import__(f"{__name__}.{name}", fromlist=[name])
        except ImportError as exc:
            raise ModuleNotFoundError(
                f"cordiag.{name} could not be imported; verify "
                f"that cordiag/{name}.py is included "
                f"(underlying error: {exc})"
            ) from exc
        globals()[name] = module  # cache
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
