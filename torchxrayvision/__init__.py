# Import submodules directly - they handle their own dependencies
from ._version import __version__

# Import submodules - if they fail during pip install, that's ok
# They will be available once dependencies are installed
try:
    from . import datasets
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import datasets: {e}")
    datasets = None

try:
    from . import models
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import models: {e}")
    models = None

try:
    from . import baseline_models
except ImportError as e:
    baseline_models = None

try:
    from . import autoencoders
except ImportError as e:
    autoencoders = None

__all__ = ["__version__", "datasets", "models", "baseline_models", "autoencoders"]
