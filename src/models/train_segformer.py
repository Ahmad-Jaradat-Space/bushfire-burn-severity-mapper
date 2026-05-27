"""Train SegFormer-B0 via the shared segmenter driver."""
from __future__ import annotations

import argparse

from src.models.train_segmenter import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/segformer_multiclass.yaml")
    p.add_argument("--fast-mode", action="store_true")
    args = p.parse_args()
    train(args.config, fast_mode=args.fast_mode)


if __name__ == "__main__":
    main()
