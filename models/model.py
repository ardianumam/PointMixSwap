import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn import DGCNN
from .attentive import Attention
from util import swap_pair_n


class AttentiveSwapNet(nn.Module):
    def __init__(self, cfg, output_channels=40):
        super(AttentiveSwapNet, self).__init__()
        self.n_pts = cfg.EXP.NUM_POINTS
        self.feature_extrator = DGCNN(cfg, output_channels)

        self.cls_token = nn.Parameter(torch.Tensor(1, cfg.TRAIN.N_DIV, cfg.DGCNN.EMB_DIM))
        nn.init.xavier_uniform_(self.cls_token)
        feat_dim = cfg.ATTENTION.LINEAR_EMB
        self.attn1 = Attention(dim_Q=cfg.DGCNN.EMB_DIM, dim_K=feat_dim, dim_LIN=cfg.DGCNN.EMB_DIM, num_heads=1, n_pts=cfg.EXP.NUM_POINTS)
        self.attn2 = Attention(dim_Q=feat_dim, dim_K=cfg.DGCNN.EMB_DIM, dim_LIN=cfg.DGCNN.EMB_DIM, num_heads=1, n_pts=cfg.EXP.NUM_POINTS)

        self.linear1 = nn.Linear(cfg.DGCNN.EMB_DIM * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=cfg.DGCNN.DROPOUT)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=cfg.DGCNN.DROPOUT)
        self.linear3 = nn.Linear(256, output_channels)

        self.encoder = nn.ModuleList()
        for enc_layer_idx in range(cfg.ATTENTION.ENCODER_N_LAYER):
            if enc_layer_idx == 0:
                self.encoder.append(Attention(dim_Q=cfg.DGCNN.EMB_DIM, dim_K=cfg.DGCNN.EMB_DIM, dim_LIN=cfg.ATTENTION.LINEAR_EMB,
                                              num_heads=cfg.ATTENTION.N_HEAD, n_pts=cfg.EXP.NUM_POINTS))
            else:
                self.encoder.append(Attention(dim_Q=cfg.ATTENTION.LINEAR_EMB, dim_K=cfg.ATTENTION.LINEAR_EMB, dim_LIN=cfg.ATTENTION.LINEAR_EMB,
                                              num_heads=cfg.ATTENTION.N_HEAD, n_pts=cfg.EXP.NUM_POINTS))

    def forward(self, x, is_return_attn=False, is_feature_mixup_turn=False, given_attention=None):
        batch_size = x.size(0)
        # backbone extractor
        x_feat = self.feature_extrator(x)  # (BS, emb_dims, num_points)

        # encoder
        x_feat = x_feat.permute(0, 2, 1)  # [BS, n_pts, emb_dim]
        for enc_layer in self.encoder:
            x_feat = enc_layer(Q=x_feat, K=x_feat, return_attn=False)

        # decoder
        h, attn1 = self.attn1(Q=self.cls_token.repeat(x_feat.size(0), 1, 1), # Q = [BS, 2, emb_dim = dim_Q]; K = [BS, n_pts, emb_dim= dim_K]
                              K=x_feat) #h = [BS,2,n_pts=1024]; attn1 = [BS,2,n_pts=1024]
        x_out, attn2 = self.attn2(Q=x_feat, K=h)  # x = [BS, n_pts, emb_dim]; attn2 = [BS,n_pts,2]
        attention = attn1

        # mixswap
        if is_feature_mixup_turn:
            if given_attention == None:
                raise Exception("Given attention must not None!")
            x_out = swap_pair_n(data=x_out,
                                attn=given_attention) # x input = [BS, n_pts, emb_dim]

        # downstream classification network
        x = x_out.permute(0,2,1) # [BS, emb_dim, n_pts]
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        if is_return_attn:
            return x, attention
        else:
            return x, None