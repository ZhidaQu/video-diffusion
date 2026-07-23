"""Anomaly-aware video restoration: detect anomaly, mask, inpaint, clean video."""

__version__ = "0.1.0"

import os as _os

# Keep HuggingFace downloads inside the project; set before transformers/diffusers import.
_weights = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "weights")
_os.environ.setdefault("HF_HOME", _os.path.join(_weights, "hf"))
_os.environ.setdefault("TORCH_HOME", _os.path.join(_weights, "torch"))

from . import registry  # noqa: F401,E402

# Import for side-effect component registration; heavy deps stay lazy.
from . import data, detect, inpaint, score  # noqa: F401,E402
