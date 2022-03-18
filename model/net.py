import megengine as mge
import megengine.module as nn
import megengine.functional as F
from model.module import Encoder, Fusion, Regression
from common import quaternion
import math


class FINet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.num_iter = params.titer
        self.net_config = params.net_config
        self.encoder = [Encoder(self.net_config) for _ in range(self.num_iter)]
        self.fusion = [Fusion() for _ in range(self.num_iter)]
        self.regression = [Regression(self.net_config) for _ in range(self.num_iter)]

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.msra_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init.calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            # elif isinstance(m, nn.BatchNorm1d):
            #     nn.init.ones_(m.weight)
            #     nn.init.zeros_(m.bias)

    def forward(self, data):
        endpoints = {}

        xyz_src = data["points_src"][:, :, :3]
        xyz_ref = data["points_ref"][:, :, :3]
        transform_gt = data["transform_gt"]
        pose_gt = data["pose_gt"]

        # init endpoints
        all_R_feats = []
        all_t_feats = []
        all_dropout_R_feats = []
        all_dropout_t_feats = []
        all_transform_pair = []
        all_pose_pair = []

        # init params
        B = xyz_src.shape[0]
        init_quat = F.tile(mge.tensor([1, 0, 0, 0], dtype="float32"), (B, 1))  # (B, 4)
        init_translate = F.tile(mge.tensor([0, 0, 0], dtype="float32"), (B, 1))  # (B, 3)
        pose_pred = F.concat((init_quat, init_translate), axis=1)  # (B, 7)

        # rename xyz_src
        xyz_src_iter = F.copy(xyz_src, device=xyz_src.device)

        for i in range(self.num_iter):
            # encoder
            encoder = self.encoder[i]
            enc_input = F.concat((xyz_src_iter.transpose(0, 2, 1).detach(), xyz_ref.transpose(0, 2, 1)), axis=0)  # 2B, C, N
            enc_feats = encoder(enc_input)
            src_enc_feats = [feat[:B, ...] for feat in enc_feats]
            ref_enc_feats = [feat[B:, ...] for feat in enc_feats]
            enc_src_R_feat = src_enc_feats[0]  # B, C
            enc_src_t_feat = src_enc_feats[1]  # B, C
            enc_ref_R_feat = ref_enc_feats[0]  # B, C
            enc_ref_t_feat = ref_enc_feats[1]  # B, C

            # GFI
            src_R_cat_feat = F.concat((enc_src_R_feat, enc_ref_R_feat), axis=-1)  # B, 2C
            ref_R_cat_feat = F.concat((enc_ref_R_feat, enc_src_R_feat), axis=-1)  # B, 2C
            src_t_cat_feat = F.concat((enc_src_t_feat, enc_ref_t_feat), axis=-1)  # B, 2C
            ref_t_cat_feat = F.concat((enc_ref_t_feat, enc_src_t_feat), axis=-1)  # B, 2C
            fusion_R_input = F.concat((src_R_cat_feat, ref_R_cat_feat), axis=0)  # 2B, C
            fusion_t_input = F.concat((src_t_cat_feat, ref_t_cat_feat), axis=0)  # 2B, C
            fusion_feats = self.fusion[i](fusion_R_input, fusion_t_input)
            src_fusion_feats = [feat[:B, ...] for feat in fusion_feats]
            ref_fusion_feats = [feat[B:, ...] for feat in fusion_feats]
            src_R_feat = src_fusion_feats[0]  # B, C
            src_t_feat = src_fusion_feats[1]  # B, C
            ref_R_feat = ref_fusion_feats[0]  # B, C
            ref_t_feat = ref_fusion_feats[1]  # B, C

            # R feats
            if self.net_config["reg_R_feats"] == "tr-tr":
                R_feats = F.concat((src_t_feat, src_R_feat, ref_t_feat, ref_R_feat), axis=-1)  # B, 4C

            elif self.net_config["reg_R_feats"] == "tr-r":
                R_feats = F.concat((src_R_feat, src_t_feat, ref_R_feat), axis=-1)  # B, 3C

            elif self.net_config["reg_R_feats"] == "r-r":
                R_feats = F.concat((src_R_feat, ref_R_feat), axis=-1)  # B, 2C

            else:
                raise ValueError("Unknown reg_R_feats order {}".format(self.net_config["reg_R_feats"]))

            # t feats
            if self.net_config["reg_t_feats"] == "tr-t":
                src_t_feats = F.concat((src_t_feat, src_R_feat, ref_t_feat), axis=-1)  # B, 3C
                ref_t_feats = F.concat((ref_t_feat, ref_R_feat, src_t_feat), axis=-1)  # B, 3C

            elif self.net_config["reg_t_feats"] == "t-t":
                src_t_feats = F.concat((src_t_feat, ref_t_feat), axis=-1)  # B, 2C
                ref_t_feats = F.concat((ref_t_feat, src_t_feat), axis=-1)  # B, 2C

            else:
                raise ValueError("Unknown reg_t_feats order {}".format(self.net_config["reg_t_feats"]))

            # regression
            t_feats = F.concat((src_t_feats, ref_t_feats), axis=0)  # 2B, 3C or 2B, 2C
            pred_quat, pred_center = self.regression[i](R_feats, t_feats)
            src_pred_center, ref_pred_center = F.split(pred_center, 2, axis=0)
            pred_translate = ref_pred_center - src_pred_center
            pose_pred_iter = F.concat((pred_quat, pred_translate), axis=-1)  # B, 7

            # extract features for compute transformation sensitivity loss (TSL)
            xyz_src_rotated = quaternion.mge_quat_rotate(xyz_src_iter.detach(), pose_pred_iter.detach())  # B, N, 3
            xyz_src_translated = xyz_src_iter.detach() + F.expand_dims(pose_pred_iter.detach()[:, 4:], axis=1)  # B, N, 3

            rotated_enc_input = F.concat((xyz_src_rotated.transpose(0, 2, 1).detach(), xyz_ref.transpose(0, 2, 1)), axis=0)  # 2B, C, N
            rotated_enc_feats = encoder(rotated_enc_input)
            rotated_src_enc_feats = [feat[:B, ...] for feat in rotated_enc_feats]
            rotated_enc_src_R_feat = rotated_src_enc_feats[0]  # B, C
            rotated_enc_src_t_feat = rotated_src_enc_feats[1]  # B, C

            translated_enc_input = F.concat((xyz_src_translated.transpose(0, 2, 1).detach(), xyz_ref.transpose(0, 2, 1)),
                                            axis=0)  # 2B, C, N
            translated_enc_feats = encoder(translated_enc_input)
            translated_src_enc_feats = [feat[:B, ...] for feat in translated_enc_feats]
            translated_enc_src_R_feat = translated_src_enc_feats[0]  # B, C
            translated_enc_src_t_feat = translated_src_enc_feats[1]  # B, C

            # dropout
            dropout_src_R_feat = src_enc_feats[2]  # B, C
            dropout_src_t_feat = src_enc_feats[3]  # B, C
            dropout_ref_R_feat = ref_enc_feats[2]  # B, C
            dropout_ref_t_feat = ref_enc_feats[3]  # B, C

            # do transform
            xyz_src_iter = quaternion.mge_quat_transform(pose_pred_iter, xyz_src_iter.detach())
            pose_pred = quaternion.mge_transform_pose(pose_pred.detach(), pose_pred_iter)
            transform_pred = quaternion.mge_quat2mat(pose_pred)

            # add endpoints at each iteration
            all_R_feats.append([enc_src_R_feat, rotated_enc_src_R_feat, translated_enc_src_R_feat])
            all_t_feats.append([enc_src_t_feat, rotated_enc_src_t_feat, translated_enc_src_t_feat])
            all_dropout_R_feats.append([dropout_src_R_feat, enc_src_R_feat, dropout_ref_R_feat, enc_ref_R_feat])
            all_dropout_t_feats.append([dropout_src_t_feat, enc_src_t_feat, dropout_ref_t_feat, enc_ref_t_feat])
            all_transform_pair.append([transform_gt, transform_pred])
            all_pose_pair.append([pose_gt, pose_pred])

            mge.coalesce_free_memory()

        # add endpoints finally
        endpoints["all_R_feats"] = all_R_feats
        endpoints["all_t_feats"] = all_t_feats
        endpoints["all_dropout_R_feats"] = all_dropout_R_feats
        endpoints["all_dropout_t_feats"] = all_dropout_t_feats
        endpoints["all_transform_pair"] = all_transform_pair
        endpoints["all_pose_pair"] = all_pose_pair
        endpoints["transform_pair"] = [transform_gt, transform_pred]
        endpoints["pose_pair"] = [pose_gt, pose_pred]

        return endpoints


def fetch_net(params):
    if params.net_type == "finet":
        net = FINet(params)

    else:
        raise NotImplementedError
    return net
