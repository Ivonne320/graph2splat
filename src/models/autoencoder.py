import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import LazyLinear

from configs import AutoencoderConfig
from src.models.backbones import (
    CrossAttention3D,
    MultiModalEncoder,
    SparseStructureDecoder,
)
from src.modules.sparse.basic import SparseTensor
from src.modules.sparse.linear import SparseLinear
from src.models.gnn_refine import SceneGraphRefiner

_LOGGER = logging.getLogger(__name__)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        cfg: AutoencoderConfig,
        device: str = "cuda",
        text_guidance: Any = None,
    ) -> None:
        super(AutoEncoder, self).__init__()
        self.cfg = cfg
        self.out_feature_dim = cfg.encoder.voxel.out_feature_dim
        self.channels = cfg.encoder.voxel.channels

        # # gnn module
 
        # # self.gnn = SceneGraphRefiner(embed_dim=cfg.encoder.global_descriptor_dim)
        # self.gnn = SceneGraphRefiner(embed_dim=512)
        # out_features = cfg.encoder.voxel.out_feature_dim
        # final_res = cfg.encoder.voxel.in_channel_res // 2 ** (len(cfg.encoder.voxel.channels) - 1)
        # decoder_input_dim = out_features * final_res ** 3
        
        # # self.gnn_to_decoder_proj = nn.Linear(
        # #     in_features=cfg.encoder.global_descriptor_dim,
        # #     out_features=decoder_input_dim,
        # # )

        # self.gnn_to_decoder_proj = nn.Sequential(
        #     # nn.Linear(cfg.encoder.global_descriptor_dim, 128),     # ~262K
        #     nn.Linear(512, 1024), 
        #     nn.ReLU(),
        #     nn.Linear(1024, decoder_input_dim)    # ~33.5M
        # )

        # # self.embedding_proj = nn.Linear(
        # #     in_features=decoder_input_dim,
        # #     # out_features=cfg.encoder.global_descriptor_dim,
        # #     out_features = 512
        # # )
        # self.embedding_proj = nn.Sequential(
        #     nn.Linear(decoder_input_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        # )
        # self.gnn_ln = nn.LayerNorm(512).to(device)

        self.encoder = MultiModalEncoder(
            modules=cfg.encoder.modules,
            rel_dim=cfg.encoder.rel_dim,
            attr_dim=cfg.encoder.attr_dim,
            img_emb_dim=cfg.encoder.img_emb_dim,
            img_feat_dim=cfg.encoder.img_patch_feat_dim,
            dropout=cfg.encoder.other.drop,
            img_aggregation_mode=getattr(cfg.encoder, "multi_view_aggregator", None),
            use_pos_enc=getattr(cfg.encoder, "use_pos_enc", False),
            in_latent_dim=cfg.encoder.voxel.in_feature_dim,
            out_latent_dim=cfg.encoder.voxel.out_feature_dim,
            voxel_encoder=cfg.decoder.net,
            channels=cfg.encoder.voxel.channels,
        ).to(device)

        out_latent_dim = cfg.encoder.voxel.out_feature_dim
        in_latent_dim = cfg.encoder.voxel.in_feature_dim
        final_res = cfg.encoder.voxel.in_channel_res // 2 ** (len(self.channels) - 1)
        modules = []
        if len(cfg.encoder.modules) > 1:
            self.cross_attn = CrossAttention3D(
                context_dim=300,
                voxel_channels=out_latent_dim,
            ).cuda()
            self.unflatten = nn.Unflatten(
                1, (out_latent_dim, final_res, final_res, final_res)
            )
        modules.append(
            nn.Unflatten(1, (out_latent_dim, final_res, final_res, final_res))
        )
        modules.append(
            SparseStructureDecoder(
                out_channels=in_latent_dim,
                latent_channels=out_latent_dim,
                num_res_blocks=1,
                channels=self.channels,
            )
        )
        self.voxel_decoder = nn.Sequential(*modules).to(device)

    # def initialize_lazy_modules(self, embedding: torch.Tensor, edge_index: torch.Tensor) -> None:
    #         """
    #         Dummy forward to initialize LazyLinear modules.
    #         """
    #         # Run through embedding_proj
    #         projected_embedding = self.embedding_proj(embedding)

    #         # Run through GNN
    #         embedding_refined = self.gnn(projected_embedding, edge_index)

    #         # Run through gnn_to_decoder_proj
    #         _ = self.gnn_to_decoder_proj(embedding_refined)
    
    def encode(self, x):
        embs = self.encoder(x)
        return embs["joint"]

    def decode(self, code) -> torch.Tensor:
        structured_latent = code
        if len(self.cfg.encoder.modules) > 1:
            voxel = self.unflatten(structured_latent[:, :-300])
            context = structured_latent[:, -300:]
            structured_latent = self.cross_attn(voxel, context).reshape(
                voxel.shape[0], -1
            )
        structured_latent = self.voxel_decoder(structured_latent)
        return structured_latent
