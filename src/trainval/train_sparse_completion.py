"""
Training pipeline for the sparse-encoder + dense-decoder completion model.

Inherits *everything* from train_unet_slat_completion.Trainer and only
overrides create_model() to swap in SparseCompletionModel.

Usage (same flags as the UNet pipeline):
    python -m src.trainval.train_sparse_completion \
        --config scripts/train_val/train_structure.yaml [--generalization ...]
"""

import logging
import os.path as osp

import torch

from configs import Config, update_configs
from src.models.sparse_completion import SparseCompletionModel
from src.trainval.train_unet_slat_completion import Trainer as _UNetTrainer
from src.trainval.train_unet_slat_completion import parse_args
from utils import common


class Trainer(_UNetTrainer):
    """
    Identical to the UNet trainer, but uses SparseCompletionModel instead of
    UNetCompletionModel.  Because create_model() is called via self inside
    _UNetTrainer.__init__(), Python's dynamic dispatch automatically routes
    model construction here without any other changes.
    """

    def create_model(self) -> SparseCompletionModel:
        out_channels = 1 + self.latent_dim
        model = SparseCompletionModel(
            feat_in=self.seed_feat_dim,
            out_channels=out_channels,
            use_instance_norm=self.use_generalization,
            dropout_p=self._gen_model_dropout_p,
        ).to(self.device)

        snapshot = getattr(self.cfg.train, "sparse_completion_snapshot", None)
        if snapshot and osp.exists(snapshot):
            state = torch.load(snapshot, map_location=self.device)
            model_state = state.get("model", state)
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            self.logger.info(
                f"Loaded sparse completion snapshot {snapshot} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        return model


def main() -> None:
    common.init_log(level=logging.INFO)
    parser, args, unknown = parse_args()
    cfg = update_configs(args.config, unknown)
    trainer = Trainer(cfg, parser)
    trainer.run()


if __name__ == "__main__":
    main()
