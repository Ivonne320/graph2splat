import logging
from torch.nn.parameter import UninitializedParameter

from configs import update_configs
from src.models.latent_autoencoder_rgb_skip import LatentAutoencoderRgbSkip
from src.trainval.train_scene_gs import Trainer as BaseTrainer, parse_args
from utils import common
from src.models.losses.reconstruction import LPIPS


class TrainerRgbSkip(BaseTrainer):
    def create_model(self) -> LatentAutoencoderRgbSkip:
        model = LatentAutoencoderRgbSkip(cfg=self.cfg.autoencoder, device=self.device)
        self.perceptual_loss = LPIPS()
        message: str = "Model created (RGB skip)"
        self.logger.info(message)
        num_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter)
        )
        self.logger.info(f"Number of parameters: {num_params}")
        return model


def main() -> None:
    """Run training with RGB skip injection."""

    common.init_log(level=logging.INFO)
    parser, args, unknown_args = parse_args()
    cfg = update_configs(args.config, unknown_args)
    trainer = TrainerRgbSkip(cfg, parser)
    trainer.run()


if __name__ == "__main__":
    main()
