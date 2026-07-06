"""Models package exports.

Expose common model classes and factory for convenience when importing
as a package (e.g. `import models; models.GatedFusionFormer`).
"""

from .model import GatedFusionFormer  # primary GFF implementation
from .cnn2 import CNN2
from .mod_rec_net import ModRecNet
from .factory import build_model

__all__ = [
    "GatedFusionFormer",
    "CNN2",
    "ModRecNet",
    "build_model",
]
