import torch
from .cnn2 import CNN2
from .mod_rec_net import ModRecNet
from .model import GatedFusionFormer


def build_model(model_name: str, config: dict):
    """
    Create and return a model instance by name.

    Supported names (case-insensitive): 'cnn2', 'modrecnet', 'gff_v3'
    """
    registry = {
        'cnn2': CNN2,
        'modrecnet': ModRecNet,
        'gffnn': GFFNN,
    }

    key = model_name.lower()
    if key not in registry:
        raise ValueError(f"Model [{model_name}] not registered. Available: {list(registry.keys())}")

    return registry[key](**config)