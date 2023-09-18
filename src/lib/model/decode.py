from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import math
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
from .utils import _nms, _Gauss_nms, _topk, _topk_channel
from utils.image import gaussian_radius
from sklearn import cluster
import numpy as np


def _update_kps_with_hm(
        kps, output, batch, num_joints, K, bboxes=None, scores=None):
    if 'hm_hp' in output:
        hm_hp = output['hm_hp']
        hm_hp = _nms(hm_hp)
        thresh = 0.2
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if 'hp_offset' in output or 'reg' in output:
            hp_offset = output['hp_offset'] if 'hp_offset' in output \
                else output['reg']
            hp_offset = _tranpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        mask = (hm_score < thresh)

        if bboxes is not None:
            l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                   (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
        else:
            l = kps[:, :, :, 0:1].min(dim=1, keepdim=True)[0]
            t = kps[:, :, :, 1:2].min(dim=1, keepdim=True)[0]
            r = kps[:, :, :, 0:1].max(dim=1, keepdim=True)[0]
            b = kps[:, :, :, 1:2].max(dim=1, keepdim=True)[0]
            margin = 0.25
            l = l - (r - l) * margin
            r = r + (r - l) * margin
            t = t - (b - t) * margin
            b = b + (b - t) * margin
            mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                   (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + mask
            # sc = (kps[:, :, :, :].max(dim=1, keepdim=True) - kps[:, :, :, :].min(dim=1))
        # mask = mask + (min_dist > 10)
        mask = (mask > 0).float()
        kps_score = (1 - mask) * hm_score + mask * \
                    scores.unsqueeze(-1).expand(batch, num_joints, K, 1)  # bJK1
        kps_score = scores * kps_score.mean(dim=1).view(batch, K)
        # kps_score[scores < 0.1] = 0
        mask = mask.expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
        return kps, kps_score
    else:
        return kps, kps


def generic_decode(output, K=100, opt=None, leiji_htmap=None):
    if not ('hm' in output):
        return {}

    if opt.zero_tracking:
        output['tracking'] *= 0

    heat = output['hm']
    heat2 = heat

    batch, cat, height, width = heat.size()

    if opt.leiji_test:
        # heat[heat < opt.track_thresh]=0
        if leiji_htmap is None:
            leiji_htmap = heat
        else:
            heat = 0.3*leiji_htmap + 0.7*heat

    # torch.save(heat, "../exp/heat_pre_test59.pt")
    if opt.nms_type == "def_conv":
        heat = _Gauss_nms(heat)
        temp_wh = output['wh']
        temp_wh[temp_wh < 0] = 0
        idx = (heat > opt.out_thresh).nonzero()
        diame = torch.zeros(heat.shape).to(heat.device)
        kern = set()
        for id_pair in idx:
            a,b,c,d =id_pair[0], id_pair[1], id_pair[2], id_pair[3]
            w, h = 4*temp_wh[a,0,c,d], 4*temp_wh[a,1,c,d]
            dia = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))/2.0))
            if dia>0 and dia % 2 == 0:
                dia = dia + 1
            if dia > 1:
                diame[a,b,c,d] = dia
                kern.add(dia)
        for k in kern:
            heat_out = _nms(heat, kernel=k)
            heat[diame==k] = heat_out[diame==k]

    elif opt.nms_type == "none":
        pass
    else:
        heat = _nms(heat)
        # if opt.leiji_test:
        #     heat2 = _nms(heat2)
        # else:
        #     heat2 = heat
    if opt.auto_thresh:
        temp_heat = (255 * torch.sum(heat, dim=1).squeeze().cpu().numpy()).astype("uint8")
        temp_X = temp_heat.flatten(order='C')
        temp_X = temp_X[temp_X > 255*0.1].reshape((-1,1))
        if temp_X.size <= 1:
            new_thresh = 0.7
        else:
            # init_p = np.array([[255*0.7], [255*0.2]])
            # cluster = KMeans(n_clusters=2, max_iter=10, init=init_p, n_init=1).fit(temp_X)
            mycluster = cluster.KMeans(n_clusters=2, max_iter=100, random_state=1).fit(temp_X)
            pre = mycluster.fit_predict(temp_X)
            if temp_X[pre==1].size > 0 and temp_X[pre==0].size > 0:
                up_thresh = max(temp_X[pre==1].min(), temp_X[pre==0].min()) / 255.0
                low_thresh = min(temp_X[pre==1].max(), temp_X[pre==0].max()) / 255.0
                if low_thresh < 0.5 and up_thresh - low_thresh > 0.1:
                    new_thresh = max(0.4, (up_thresh + low_thresh)/2.0)
                else:
                    new_thresh = 0.4
            else:
                new_thresh = 0.6

    # torch.save(heat, "../exp/heat_aft_test1.pt")
    scores, inds, clses, ys0, xs0 = _topk(heat, K=K)

    clses = clses.view(batch, K)
    scores = scores.view(batch, K)
    bboxes = None
    cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
    ret = {'scores': scores, 'clses': clses.float(),
           'xs': xs0, 'ys': ys0, 'cts': cts}
    if 'reg' in output:
        reg = output['reg']
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs0.view(batch, K, 1) + 0.5
        ys = ys0.view(batch, K, 1) + 0.5

    if 'wh' in output:
        wh = output['wh']
        wh = _tranpose_and_gather_feat(wh, inds)  # B x K x (F)
        # wh = wh.view(batch, K, -1)
        wh = wh.view(batch, K, 2)
        wh[wh < 0] = 0
        if wh.size(2) == 2 * cat:  # cat spec
            wh = wh.view(batch, K, -1, 2)
            cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
            wh = wh.gather(2, cats.long()).squeeze(2)  # B x K x 2
        else:
            pass
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        ret['bboxes'] = bboxes
        # print('ret bbox', ret['bboxes'])

    if 'ltrb' in output:
        ltrb = output['ltrb']
        ltrb = _tranpose_and_gather_feat(ltrb, inds)  # B x K x 4
        ltrb = ltrb.view(batch, K, 4)
        bboxes = torch.cat([xs0.view(batch, K, 1) + ltrb[..., 0:1],
                            ys0.view(batch, K, 1) + ltrb[..., 1:2],
                            xs0.view(batch, K, 1) + ltrb[..., 2:3],
                            ys0.view(batch, K, 1) + ltrb[..., 3:4]], dim=2)
        ret['bboxes'] = bboxes

    regression_heads = ['tracking', 'dep', 'rot', 'dim', 'amodel_offset',
                        'nuscenes_att', 'velocity']

    for head in regression_heads:
        if head in output:
            ret[head] = _tranpose_and_gather_feat(
                output[head], inds).view(batch, K, -1)

    if 'ltrb_amodal' in output:
        ltrb_amodal = output['ltrb_amodal']
        ltrb_amodal = _tranpose_and_gather_feat(ltrb_amodal, inds)  # B x K x 4
        ltrb_amodal = ltrb_amodal.view(batch, K, 4)
        bboxes_amodal = torch.cat([xs0.view(batch, K, 1) + ltrb_amodal[..., 0:1],
                                   ys0.view(batch, K, 1) + ltrb_amodal[..., 1:2],
                                   xs0.view(batch, K, 1) + ltrb_amodal[..., 2:3],
                                   ys0.view(batch, K, 1) + ltrb_amodal[..., 3:4]], dim=2)
        ret['bboxes_amodal'] = bboxes_amodal
        ret['bboxes'] = bboxes_amodal

    if 'hps' in output:
        kps = output['hps']
        num_joints = kps.shape[1] // 2
        kps = _tranpose_and_gather_feat(kps, inds)
        kps = kps.view(batch, K, num_joints * 2)
        kps[..., ::2] += xs0.view(batch, K, 1).expand(batch, K, num_joints)
        kps[..., 1::2] += ys0.view(batch, K, 1).expand(batch, K, num_joints)
        kps, kps_score = _update_kps_with_hm(
            kps, output, batch, num_joints, K, bboxes, scores)
        ret['hps'] = kps
        ret['kps_score'] = kps_score

    if 'pre_inds' in output and output['pre_inds'] is not None:
        pre_inds = output['pre_inds']  # B x pre_K
        pre_K = pre_inds.shape[1]
        pre_ys = (pre_inds / width).int().float()
        pre_xs = (pre_inds % width).int().float()

        ret['pre_cts'] = torch.cat(
            [pre_xs.unsqueeze(2), pre_ys.unsqueeze(2)], dim=2)
    if opt.atten_method != "none":
        if opt.leiji_test:
            return ret, heat2, leiji_htmap
        if opt.auto_thresh:
            return ret, heat2, new_thresh
        return ret, heat2, output['tracking']
    return ret
