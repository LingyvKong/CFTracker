from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar

from model.data_parallel import DataParallel
from utils.utils import AverageMeter

from model.losses import FastFocalLoss, RegWeightedL1Loss
from model.losses import BinRotLoss, WeightedBCELoss
from model.decode import generic_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr
from utils.debugger import Debugger
from utils.post_process import generic_post_process
from torch.autograd import Variable
from min_norm_solvers import MinNormSolver, gradient_normalizers
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

class GenericLoss(torch.nn.Module):
    def __init__(self, opt):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss(opt=opt)
        self.crit_reg = RegWeightedL1Loss()
        if 'rot' in opt.heads:
            self.crit_rot = BinRotLoss()
        if 'nuscenes_att' in opt.heads:
            self.crit_nuscenes_att = WeightedBCELoss()
        self.opt = opt

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        if 'hm_hp' in output:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        if 'dep' in output:
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        return output

    def forward(self, outputs, batch, head=None):
        opt = self.opt
        # scale_pred = F.upsample(input=batch['hm'], size=(608, 992), mode='bilinear', align_corners=True)
        # scale_pred = torch.squeeze(scale_pred, 0)
        # scale_pred = torch.mean(scale_pred, dim=0)
        # visual = scale_pred.cpu().numpy()
        # fig = plt.gcf()
        # fig.set_size_inches(2, 2)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # im = torch.squeeze(batch['image'], 0)
        # im_v = im.permute(1, 2, 0).cpu().numpy()
        # plt.imshow(im_v)
        # plt.imshow(visual, alpha=0.5, cmap='jet')
        # plt.axis('off')
        # path = '/workspace/debug/'+ '000001.jpg'
        # path_p = path[0:-11]
        # if not os.path.exists(path_p):
        #     os.makedirs(path_p)
        # plt.savefig(path, dpi=1000)

        if opt.mtl_loss:
            if head == "hm":
                outputs = _sigmoid(outputs)
                loss_h = self.crit(outputs, batch['hm'], batch['ind'], batch['mask'], batch['cat'])
            else:
                loss_h = self.crit_reg(outputs, batch[head + '_mask'], batch['ind'], batch[head])
            return loss_h
        else:
            losses = {head: 0 for head in opt.heads}

            for s in range(opt.num_stacks):
                output = outputs[s]
                output = self._sigmoid_output(output)

                if 'hm' in output:
                    losses['hm'] += self.crit(
                        output['hm'], batch['hm'], batch['ind'],
                        batch['mask'], batch['cat']) / opt.num_stacks

                regression_heads = [
                    'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
                    'dep', 'dim', 'amodel_offset', 'velocity']

                for head in regression_heads:
                    if head in output:
                        losses[head] += self.crit_reg(
                            output[head], batch[head + '_mask'],
                            batch['ind'], batch[head]) / opt.num_stacks

                if 'hm_hp' in output:
                    losses['hm_hp'] += self.crit(
                        output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
                        batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
                    if 'hp_offset' in output:
                        losses['hp_offset'] += self.crit_reg(
                            output['hp_offset'], batch['hp_offset_mask'],
                            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

                if 'rot' in output:
                    losses['rot'] += self.crit_rot(
                        output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
                        batch['rotres']) / opt.num_stacks

                if 'nuscenes_att' in output:
                    losses['nuscenes_att'] += self.crit_nuscenes_att(
                        output['nuscenes_att'], batch['nuscenes_att_mask'],
                        batch['ind'], batch['nuscenes_att']) / opt.num_stacks

            losses['tot'] = 0
            for head in opt.heads:
                losses['tot'] += opt.weights[head] * losses[head]

            return losses['tot'], losses


class ModleWithLoss(torch.nn.Module):
    def __init__(self, opt, model, loss):
        super(ModleWithLoss, self).__init__()
        self.opt = opt
        self.model = model
        self.loss = loss

    def forward(self, batch, head="", rep=None):
        if not self.opt.mtl_loss:
            pre_img = batch['pre_img'] if 'pre_img' in batch else None
            pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
            outputs = self.model(batch['image'], pre_img, pre_hm)
            loss, loss_stats = self.loss(outputs, batch)
            return outputs[-1], loss, loss_stats
        else:
            if head == "encode_1":
                with torch.no_grad():
                    pre_img_volatile = Variable(batch['pre_img']) if 'pre_img' in batch else None
                    pre_hm_volatile = Variable(batch['pre_hm']) if 'pre_hm' in batch else None
                    img_volatile = Variable(batch['image'])
                rep = self.model.forward_encode(img_volatile, pre_img_volatile, pre_hm_volatile)
                return rep
            elif head == "encode_2":
                pre_img = batch['pre_img'] if 'pre_img' in batch else None
                pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
                rep = self.model.forward_encode(batch['image'], pre_img, pre_hm)
                return rep
            else:
                out_head = self.model.forward_decode(head, rep)
                loss_h = self.loss(out_head, batch, head)
                return out_head, loss_h


class Trainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(opt, model, self.loss)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats \
                          if l == 'tot' or opt.weights[l] > 0}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)
            if not self.opt.mtl_loss:
                output, loss, loss_stats = model_with_loss(batch)
                loss = loss.mean()
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                if phase == 'train':
                    loss_stats = {}
                    output = {}
                    # loss_h_dict = {}
                    loss_data = {}
                    grads = {}
                    scale = {}
                    self.optimizer.zero_grad()
                    rep = model_with_loss(batch, head='encode_1')
                    rep_variable = Variable(rep.data.clone(), requires_grad=True)
                    for head in self.opt.heads:
                        self.optimizer.zero_grad()
                        out_head, loss_h = model_with_loss(batch, head, rep=rep_variable)
                        # output[head] = out_head
                        loss_h = loss_h.mean()
                        # loss_h_dict[head] = loss_h
                        loss_data[head] = loss_h.data
                        loss_h.backward()
                        grads[head] = []
                        grads[head].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                        rep_variable.grad.data.zero_()
                    gn = gradient_normalizers(grads, loss_data, "loss+")
                    for head in self.opt.heads:
                        for gr_i in range(len(grads[head])):
                            grads[head][gr_i] = grads[head][gr_i] / gn[head]
                    sol, min_norm = MinNormSolver.find_min_norm_element([grads[h] for h in self.opt.heads])
                    for i, t in enumerate(self.opt.heads):
                        scale[t] = float(sol[i])

                    self.optimizer.zero_grad()
                    if self.opt.freeze_encoder:
                        rep = rep_variable
                    else:
                        rep = model_with_loss(batch, head="encode_2")
                    # for i, head in enumerate(self.opt.heads):
                    #     if i > 0:
                    #         loss = loss + scale[head] * loss_h_dict[head]
                    #     else:
                    #         loss = scale[head] * loss_h_dict[head]
                    #     loss_stats[head] = scale[head] * loss_h_dict[head]
                    # loss_stats['tot'] = loss
                else:
                    rep = model_with_loss(batch, head="encode_2")
                for i, head in enumerate(self.opt.heads):
                    out_head, loss_h = model_with_loss(batch, head, rep)
                    output[head] = out_head
                    loss_h = loss_h.mean()
                    loss_data[head] = loss_h.data
                    if phase == 'train':
                        if i > 0:
                            loss = loss + scale[head]*loss_h
                        else:
                            loss = scale[head]*loss_h
                        loss_stats[head] = scale[head]*loss_h
                    else:
                        if i > 0:
                            loss = loss + loss_h
                        else:
                            loss = loss_h
                        loss_stats[head] = loss_h
                loss_stats['tot'] = loss
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['image'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if self.opt.mtl_loss:
                for h in scale:
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format('scale-'+h,scale[h])
            if opt.print_iter > 0:  # If not using progress bar
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.debug >=1:
                self.debug(batch, output, iter_id, dataset=data_loader.dataset)

            del output, loss, loss_stats
            if self.opt.mtl_loss:
                del loss_data, grads

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        if self.opt.mtl_loss:
            for i, t in enumerate(self.opt.heads):
                ret['scale'+t] = scale[t]
        return ret, results

    def _get_losses(self, opt):
        loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
                      'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset', \
                      'ltrb_amodal', 'tracking', 'nuscenes_att', 'velocity']
        loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
        loss = GenericLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id, dataset):
        opt = self.opt
        if 'pre_hm' in batch:
            output.update({'pre_hm': batch['pre_hm']})
        dets = generic_decode(output, K=opt.K, opt=opt)
        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()
        dets_gt = batch['meta']['gt_det']
        for i in range(1):
            debugger = Debugger(opt=opt, dataset=dataset)
            img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')

            if 'pre_img' in batch:
                pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
                pre_img = np.clip(((
                                           pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
                debugger.add_img(pre_img, 'pre_img_pred')
                debugger.add_img(pre_img, 'pre_img_gt')
                if 'pre_hm' in batch:
                    pre_hm = debugger.gen_colormap(
                        batch['pre_hm'][i].detach().cpu().numpy())
                    debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

            debugger.add_img(img, img_id='out_pred')
            if 'ltrb_amodal' in opt.heads:
                debugger.add_img(img, img_id='out_pred_amodal')
                debugger.add_img(img, img_id='out_gt_amodal')

            # Predictions
            for k in range(len(dets['scores'][i])):
                if dets['scores'][i, k] > opt.vis_thresh:
                    debugger.add_coco_bbox(
                        dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
                        dets['scores'][i, k], img_id='out_pred')

                    if 'ltrb_amodal' in opt.heads:
                        debugger.add_coco_bbox(
                            dets['bboxes_amodal'][i, k] * opt.down_ratio, dets['clses'][i, k],
                            dets['scores'][i, k], img_id='out_pred_amodal')

                    if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
                        debugger.add_coco_hp(
                            dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

                    if 'tracking' in opt.heads:
                        debugger.add_arrow(
                            dets['cts'][i][k] * opt.down_ratio,
                            dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
                        debugger.add_arrow(
                            dets['cts'][i][k] * opt.down_ratio,
                            dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

            # Ground truth
            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt['scores'][i])):
                if dets_gt['scores'][i][k] > opt.vis_thresh:
                    debugger.add_coco_bbox(
                        dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
                        dets_gt['scores'][i][k], img_id='out_gt')

                    if 'ltrb_amodal' in opt.heads:
                        debugger.add_coco_bbox(
                            dets_gt['bboxes_amodal'][i, k] * opt.down_ratio,
                            dets_gt['clses'][i, k],
                            dets_gt['scores'][i, k], img_id='out_gt_amodal')

                    if 'hps' in opt.heads and \
                            (int(dets['clses'][i, k]) == 0):
                        debugger.add_coco_hp(
                            dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

                    if 'tracking' in opt.heads:
                        debugger.add_arrow(
                            dets_gt['cts'][i][k] * opt.down_ratio,
                            dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
                        debugger.add_arrow(
                            dets_gt['cts'][i][k] * opt.down_ratio,
                            dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

            if 'hm_hp' in opt.heads:
                pred = debugger.gen_colormap_hp(
                    output['hm_hp'][i].detach().cpu().numpy())
                gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
                debugger.add_blend_img(img, pred, 'pred_hmhp')
                debugger.add_blend_img(img, gt, 'gt_hmhp')

            if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
                dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
                calib = batch['meta']['calib'].detach().numpy() \
                    if 'calib' in batch['meta'] else None
                det_pred = generic_post_process(opt, dets,
                                                batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
                                                output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                                calib)
                det_gt = generic_post_process(opt, dets_gt,
                                              batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
                                              output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
                                              calib)

                debugger.add_3d_detection(
                    batch['meta']['img_path'][i], batch['meta']['flipped'][i],
                    det_pred[i], calib[i],
                    vis_thresh=opt.vis_thresh, img_id='add_pred')
                debugger.add_3d_detection(
                    batch['meta']['img_path'][i], batch['meta']['flipped'][i],
                    det_gt[i], calib[i],
                    vis_thresh=opt.vis_thresh, img_id='add_gt')
                debugger.add_bird_views(det_pred[i], det_gt[i],
                                        vis_thresh=opt.vis_thresh, img_id='bird_pred_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            elif opt.debug == 3 or opt.debug >= 5:
                debugger.show_all_imgs(pause=True)

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
