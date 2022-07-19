import torch
import torch.nn as nn


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # [BS,n_feat,n_pts)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # [BS,1,1]

    idx = idx + idx_base  # (BS, num_points, k)

    idx = idx.view(-1)  # (BS*num_points*k)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # [BS*num_points*k, num_dims]+range(0, batch_size*num_points)
    feature = feature.view(batch_size, num_points, k, num_dims)  # [BS, num_points, k, num_dims]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # [BS, num_points, k, num_dims]

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # [BS, 2*num_dims, num_points, k]

    return feature


class DGCNN(nn.Module):
    def __init__(self, cfg, output_channels=40):
        super(DGCNN, self).__init__()
        self.k = cfg.DGCNN.K

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(cfg.DGCNN.EMB_DIM)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, cfg.DGCNN.EMB_DIM, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # [BS, 2*num_dims=2*3, num_points, k]
        x = self.conv1(x)  # [BS, 64, num_points, k]
        x1 = x.max(dim=-1, keepdim=False)[0]  # [BS, 64, num_points]

        x = get_graph_feature(x1, k=self.k)  # [BS, 2*num_dims=2*64, num_points, k]
        x = self.conv2(x)  # [BS, 64, num_points, k]
        x2 = x.max(dim=-1, keepdim=False)[0]  # [BS, 64, num_points]

        x = get_graph_feature(x2, k=self.k)  # [BS, 2*num_dims=2*64, num_points, k]
        x = self.conv3(x)  # [BS, 128, num_points, k]
        x3 = x.max(dim=-1, keepdim=False)[0]  # [BS, 128, num_points, k]

        x = get_graph_feature(x3, k=self.k)  # [BS, 2*num_dims=2*128, num_points, k]
        x = self.conv4(x)  # [BS, 256, num_points, k]
        x4 = x.max(dim=-1, keepdim=False)[0]  # [BS, 256, num_points]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # [BS, 64+64+128+256=512, num_points, k]

        x_feat = self.conv5(x)  # (BS, 64+64+128+256, num_points) -> (BS, emb_dims, num_points)

        return x_feat