import json
import unittest
from pathlib import Path

from deploy.checkpoint import download_checkpoint, inspect_checkpoint, load_checkpoint_strict
from src.models.model import GatedFusionFormer


class CheckpointLoadingTests(unittest.TestCase):
    def test_private_raw_state_dict_loads_strictly(self) -> None:
        config_path = Path("deploy/config/model_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint_path = download_checkpoint(filename=config["checkpoint_filename"])
        info = inspect_checkpoint(checkpoint_path)
        model = GatedFusionFormer(
            embed_dim=config["embed_dim"],
            num_classes=config["num_classes"],
            stft_time_steps=config["stft_time_steps"],
            depth=config["depth"],
            num_heads=config["num_heads"],
        )
        loaded_info = load_checkpoint_strict(model, checkpoint_path)
        self.assertEqual(info.checkpoint_format, "raw_state_dict")
        self.assertEqual(loaded_info.sha256, config["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()