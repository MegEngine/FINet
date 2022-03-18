import numpy as np
import megengine.functional as F
from common import se3, so3


def compute_losses(endpoints, params):
    loss = {}
    # compute losses
    if params.loss_type == "finet":
        num_iter = len(endpoints["all_pose_pair"])
        triplet_loss = {}
        for i in range(num_iter):
            # reg loss
            pose_pair = endpoints["all_pose_pair"][i]
            loss["quat_{}".format(i)] = F.nn.l1_loss(pose_pair[0][:, :4], pose_pair[1][:, :4]) * params.loss_alpha1
            loss["translate_{}".format(i)] = F.nn.square_loss(pose_pair[0][:, 4:], pose_pair[1][:, 4:]) * params.loss_alpha2

            # transformation sensitivity loss (TSL)
            if i < 2:
                all_R_feats = endpoints["all_R_feats"][i]
                all_t_feats = endpoints["all_t_feats"][i]
                # R feats triplet loss
                R_feats_pos = F.nn.square_loss(all_t_feats[0], all_t_feats[1])
                R_feats_neg = F.nn.square_loss(all_R_feats[0], all_R_feats[1])
                triplet_loss["R_feats_triplet_pos_{}".format(i)] = R_feats_pos
                triplet_loss["R_feats_triplet_neg_{}".format(i)] = R_feats_neg
                loss["R_feats_triplet_{}".format(i)] = (F.clip(-R_feats_neg + params.margin[i], lower=0.0) +
                                                        R_feats_pos) * params.loss_alpha3
                # t feats triplet loss
                t_feats_pos = F.nn.square_loss(all_R_feats[0], all_R_feats[2])
                t_feats_neg = F.nn.square_loss(all_t_feats[0], all_t_feats[2])
                triplet_loss["t_feats_triplet_pos_{}".format(i)] = t_feats_pos
                triplet_loss["t_feats_triplet_neg_{}".format(i)] = t_feats_neg
                loss["t_feats_triplet_{}".format(i)] = (F.clip(-t_feats_neg + params.margin[i], lower=0.0) +
                                                        t_feats_pos) * params.loss_alpha3

            # point-wise feature dropout loss (PFDL)
            all_dropout_R_feats = endpoints["all_dropout_R_feats"][i]
            all_dropout_t_feats = endpoints["all_dropout_t_feats"][i]
            loss["src_R_feats_dropout_{}".format(i)] = F.nn.square_loss(all_dropout_R_feats[0], all_dropout_R_feats[1]) * params.loss_alpha4
            loss["ref_R_feats_dropout_{}".format(i)] = F.nn.square_loss(all_dropout_R_feats[2], all_dropout_R_feats[3]) * params.loss_alpha4
            loss["src_t_feats_dropout_{}".format(i)] = F.nn.square_loss(all_dropout_t_feats[0], all_dropout_t_feats[1]) * params.loss_alpha4
            loss["ref_t_feats_dropout_{}".format(i)] = F.nn.square_loss(all_dropout_t_feats[2], all_dropout_t_feats[3]) * params.loss_alpha4
        # total loss
        total_losses = []
        for k in loss:
            total_losses.append(loss[k])
        loss["total"] = F.sum(F.concat(total_losses))

    else:
        raise NotImplementedError
    return loss


def compute_metrics(endpoints, params):
    metrics = {}
    gt_transforms = endpoints["transform_pair"][0]
    pred_transforms = endpoints["transform_pair"][1]

    # Euler angles, Individual translation errors (Deep Closest Point convention)
    if "prnet" in params.transform_type:
        r_gt_euler_deg = so3.mge_dcm2euler(gt_transforms[:, :3, :3], seq="zyx")
        r_pred_euler_deg = so3.mge_dcm2euler(pred_transforms[:, :3, :3], seq="zyx")
    else:
        r_gt_euler_deg = so3.mge_dcm2euler(gt_transforms[:, :3, :3], seq="xyz")
        r_pred_euler_deg = so3.mge_dcm2euler(pred_transforms[:, :3, :3], seq="xyz")
    t_gt = gt_transforms[:, :3, 3]
    t_pred = pred_transforms[:, :3, 3]

    r_mse = F.mean((r_gt_euler_deg - r_pred_euler_deg)**2, axis=1)
    r_mae = F.mean(F.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
    t_mse = F.mean((t_gt - t_pred)**2, axis=1)
    t_mae = F.mean(F.abs(t_gt - t_pred), axis=1)

    r_mse = F.mean(r_mse)
    t_mse = F.mean(t_mse)
    r_mae = F.mean(r_mae)
    t_mae = F.mean(t_mae)

    # Rotation, translation errors (isotropic, i.e. doesn"t depend on error
    # direction, which is more representative of the actual error)
    concatenated = se3.mge_concatenate(se3.mge_inverse(gt_transforms), pred_transforms)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = F.acos(F.clip(0.5 * (rot_trace - 1), -1.0, 1.0)) * 180.0 / np.pi
    residual_transmag = F.norm(concatenated[:, :, 3], axis=-1)
    err_r = F.mean(residual_rotdeg)
    err_t = F.mean(residual_transmag)

    # weighted score of isotropic errors
    score = err_r * 0.01 + err_t

    metrics = {"R_MSE": r_mse, "R_MAE": r_mae, "t_MSE": t_mse, "t_MAE": t_mae, "Err_R": err_r, "Err_t": err_t, "score": score}

    return metrics
