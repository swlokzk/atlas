"""Run entrypoint for cnogb-abnormal-intervention demo pipeline.

Usage:
    python run.py --dry-run
    python run.py --run
"""
import argparse
from src.config import Config
from src.utils import set_seed


def main(args):
    cfg = Config()
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    if args.dry_run:
        print("Dry run: directories created and config:")
        print(cfg)
        return

    # TODO: wire up data -> features -> train -> evaluate
    print("Running pipeline (not implemented). Use this file as the orchestration entrypoint.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Create dirs and show config')
    parser.add_argument('--run', action='store_true', help='Execute the pipeline')
    args = parser.parse_args()
    main(args)
