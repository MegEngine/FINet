import megengine as mge
import megengine.module as nn
import megengine.functional as F


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # R
        self.R_block1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        self.R_block2 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        self.R_block3 = nn.Sequential(nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU())
        self.R_block4 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU())
        self.R_block5 = nn.Sequential(nn.Conv1d(512, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU())

        # t
        self.t_block1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        self.t_block2 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU())
        self.t_block3 = nn.Sequential(nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU())
        self.t_block4 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU())
        self.t_block5 = nn.Sequential(nn.Conv1d(512, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU())

    def forward(self, x, mask=None):
        B, C, N = x.shape
        if self.training:
            rand_mask = mge.random.uniform(size=(B, 1, N)) > self.config["dropout_ratio"]
        else:
            rand_mask = 1

        # R stage1
        R_feat_output1 = self.R_block1(x)
        if mask is not None:
            R_feat_output1 = R_feat_output1 * mask
        R_feat_output2 = self.R_block2(R_feat_output1)
        if mask is not None:
            R_feat_output2 = R_feat_output2 * mask
        R_feat_glob2 = F.max(R_feat_output2, axis=-1, keepdims=True)

        # t stage1
        t_feat_output1 = self.t_block1(x)
        if mask is not None:
            t_feat_output1 = t_feat_output1 * mask
        t_feat_output2 = self.t_block2(t_feat_output1)
        if mask is not None:
            t_feat_output2 = t_feat_output2 * mask
        t_feat_glob2 = F.max(t_feat_output2, axis=-1, keepdims=True)

        # exchange1
        src_R_feat_glob2, ref_R_feat_glob2 = F.split(R_feat_glob2, 2, axis=0)
        src_t_feat_glob2, ref_t_feat_glob2 = F.split(t_feat_glob2, 2, axis=0)
        exchange_R_feat = F.concat((F.repeat(ref_R_feat_glob2, N, axis=2), F.repeat(src_R_feat_glob2, N, axis=2)), axis=0)
        exchange_t_feat = F.concat((F.repeat(ref_t_feat_glob2, N, axis=2), F.repeat(src_t_feat_glob2, N, axis=2)), axis=0)
        exchange_R_feat = F.concat((R_feat_output2, exchange_R_feat.detach()), axis=1)
        exchange_t_feat = F.concat((t_feat_output2, exchange_t_feat.detach()), axis=1)

        # R stage2
        R_feat_output3 = self.R_block3(exchange_R_feat)
        if mask is not None:
            R_feat_output3 = R_feat_output3 * mask
        R_feat_output4 = self.R_block4(R_feat_output3)
        if mask is not None:
            R_feat_output4 = R_feat_output4 * mask
        R_feat_glob4 = F.max(R_feat_output4, axis=-1, keepdims=True)

        # t stage2
        t_feat_output3 = self.t_block3(exchange_t_feat)
        if mask is not None:
            t_feat_output3 = t_feat_output3 * mask
        t_feat_output4 = self.t_block4(t_feat_output3)
        if mask is not None:
            t_feat_output4 = t_feat_output4 * mask
        t_feat_glob4 = F.max(t_feat_output4, axis=-1, keepdims=True)

        # exchange2
        src_R_feat_glob4, ref_R_feat_glob4 = F.split(R_feat_glob4, 2, axis=0)
        src_t_feat_glob4, ref_t_feat_glob4 = F.split(t_feat_glob4, 2, axis=0)
        exchange_R_feat = F.concat((F.repeat(ref_R_feat_glob4, N, axis=2), F.repeat(src_R_feat_glob4, N, axis=2)), axis=0)
        exchange_t_feat = F.concat((F.repeat(ref_t_feat_glob4, N, axis=2), F.repeat(src_t_feat_glob4, N, axis=2)), axis=0)
        exchange_R_feat = F.concat((R_feat_output4, exchange_R_feat.detach()), axis=1)
        exchange_t_feat = F.concat((t_feat_output4, exchange_t_feat.detach()), axis=1)

        # R stage3
        R_feat_output5 = self.R_block5(exchange_R_feat)
        if mask is not None:
            R_feat_output5 = R_feat_output5 * mask

        # t stage3
        t_feat_output5 = self.t_block5(exchange_t_feat)
        if mask is not None:
            t_feat_output5 = t_feat_output5 * mask

        # final
        R_final_feat_output = F.concat((R_feat_output1, R_feat_output2, R_feat_output3, R_feat_output4, R_feat_output5), axis=1)
        t_final_feat_output = F.concat((t_feat_output1, t_feat_output2, t_feat_output3, t_feat_output4, t_feat_output5), axis=1)

        R_final_glob_feat = F.max(R_final_feat_output, axis=-1, keepdims=False)
        t_final_glob_feat = F.max(t_final_feat_output, axis=-1, keepdims=False)

        R_final_feat_dropout = R_final_feat_output * rand_mask
        R_final_feat_dropout = F.max(R_final_feat_dropout, axis=-1, keepdims=False)

        t_final_feat_dropout = t_final_feat_output * rand_mask
        t_final_feat_dropout = F.max(t_final_feat_dropout, axis=-1, keepdims=False)

        return [R_final_glob_feat, t_final_glob_feat, R_final_feat_dropout, t_final_feat_dropout]


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()

        # R
        self.R_block1 = nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU())
        self.R_block2 = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.R_block3 = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU())

        # t
        self.t_block1 = nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU())
        self.t_block2 = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.t_block3 = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU())

    def forward(self, R_feat, t_feat):
        # R
        fuse_R_feat = self.R_block1(R_feat)
        fuse_R_feat = self.R_block2(fuse_R_feat)
        fuse_R_feat = self.R_block3(fuse_R_feat)
        # t
        fuse_t_feat = self.t_block1(t_feat)
        fuse_t_feat = self.t_block2(fuse_t_feat)
        fuse_t_feat = self.t_block3(fuse_t_feat)

        return [fuse_R_feat, fuse_t_feat]


class Regression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config["reg_R_feats"] == "tr-tr":
            R_in_channel = 4096
        elif self.config["reg_R_feats"] == "tr-r":
            R_in_channel = 3072
        elif self.config["reg_R_feats"] == "r-r":
            R_in_channel = 2048
        else:
            raise ValueError("Unknown reg_R_feats order {}".format(self.config["reg_R_feats"]))

        if self.config["reg_t_feats"] == "tr-t":
            t_in_channel = 3072
        elif self.config["reg_t_feats"] == "t-t":
            t_in_channel = 2048
        else:
            raise ValueError("Unknown reg_t_feats order {}".format(self.config["reg_t_feats"]))

        self.R_net = nn.Sequential(
            # block 1
            nn.Linear(R_in_channel, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            # block 2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # block 3
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # block 4
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # final fc
            nn.Linear(256, 4),
        )

        self.t_net = nn.Sequential(
            # block 1
            nn.Linear(t_in_channel, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            # block 2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # block 3
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # block 4
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # final fc
            nn.Linear(256, 3),
        )

    def forward(self, R_feat, t_feat):

        pred_quat = self.R_net(R_feat)
        pred_quat = F.normalize(pred_quat, axis=1)
        pred_translate = self.t_net(t_feat)

        return [pred_quat, pred_translate]
