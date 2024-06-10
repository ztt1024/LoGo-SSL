import os
import time
import pandas as pd
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia_transform import aug_kornia

def train_moco(train_loader, ls_model, criterion, ls_optimizer, epoch, args, cr=0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    ls_record = [batch_time, data_time, losses, top1, top5]
    dict_result = {'Loss': [], 'Top1': [], 'Top5': []}
    if args.mc:
        losses_g = AverageMeter('Loss_g', ':.4e')
        losses_l = AverageMeter('Loss_l', ':.4e')
        ls_record.append(losses_g)
        ls_record.append(losses_l)
        dict_result['Loss_g'] = []
        dict_result['Loss_l'] = []
        if args.logo:
            losses_rmax = AverageMeter('Regressor_max', ':.4e')
            losses_rmin = AverageMeter('Regressor_min', ':.4e')
            ls_record.append(losses_rmin)
            ls_record.append(losses_rmax)
            dict_result['Regressor_max'] = []
            dict_result['Regressor_min'] = []
            regressor = ls_model[1]
            regressor.train()

            optimizer_reg = ls_optimizer[2]
            optimizer_logo = ls_optimizer[1]

    optimizer = ls_optimizer[0]
    progress = ProgressMeter(
        len(train_loader),
        ls_record,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model = ls_model[0]
    model.train()#model.eval()
    end = time.time()
    norm0 = 0

    aug = aug_kornia().cuda()

    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            for idx, img in enumerate(images):
                images[idx] = aug(img.cuda(args.gpu, non_blocking=True))

        b = images[0].shape[0]
        #q, k, _ = model(img=images[2:], local=True)
        if args.logo:
            q = model(img=images[2:], local=True)
            Vq = 0.3 / (args.fea_dim ** .5) * torch.randn(b, args.fea_dim).cuda()
            Vk = 0.3 / (args.fea_dim ** .5) * torch.randn(b, args.fea_dim).cuda()

            z_pos_12 = torch.cat([q[:b].detach() + Vq, q[b:].detach() + Vk], dim=1)
            z_pos_21 = torch.cat([q[b:].detach() + Vq, q[:b].detach() + Vk], dim=1)
            loss_rmax = .5 * regressor(z_pos_12, 'pos') + .5 * regressor(z_pos_21, 'pos')

            # randomly sample local crops from two different instances as negative pair
            z_neg_12 = torch.cat([q[:b].detach(), q[b:].detach().flip(0)], dim=1)
            z_neg_21 = torch.cat([q[b:].detach(), q[:b].detach().flip(0)], dim=1)
            loss_rmin = .5 * regressor(z_neg_12, 'neg') + .5 * regressor(z_neg_21, 'neg')

            losses_rmin.update(loss_rmin.mean().item(), b)
            dict_result['Regressor_min'].append(losses_rmin.output())

            loss_omega = (loss_rmax + loss_rmin).mean()

            optimizer_reg.zero_grad()
            loss_omega.backward()
            optimizer_reg.step()

            loss_regressor = 0

            loss_regressor += regressor(
                torch.cat([q[: b], q[b:]], dim=1), 'neg').mean()
            losses_rmax.update(loss_regressor.item(), b)

            dict_result['Regressor_max'].append(losses_rmax.output())

            optimizer_logo.zero_grad()
            loss_regressor.backward()
            norm1 = nn.utils.clip_grad_norm_(model.module.encoder_q.parameters(), cr * norm0)
            optimizer_logo.step()

        q, k, neg = model(img=images)

        target = torch.zeros(b, dtype=torch.long).cuda()
        nce = lambda q, k: nce_loss(q, k, neg, args.moco_t, target, criterion, False)
        with torch.no_grad():
            _, logits = nce_loss(q[:b].detach(), k[b:2*b],neg, args.moco_t, target, criterion, True)
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1.update(acc1[0].item(), b)
        top5.update(acc5[0].item(), b)
        dict_result['Top1'].append(top1.output())
        dict_result['Top5'].append(top5.output())

        if args.mc:
            # use multi-crop follow LoGo setting, which includes local-global and global-global pairs
            loss_lg = 0
            loss_gg = 0

            # global-global loss
            loss_gg += nce(q[:b], k[b:2 * b])
            loss_gg += nce(q[b:2 * b], k[:b])

            # local-global loss
            loss_lg += nce(q[2 * b:3 * b], k[:b]) + nce(q[2 * b:3 * b], k[b:2 * b])
            loss_lg += nce(q[3 * b:], k[:b]) + nce(q[3 * b:], k[b:2 * b])

            loss_nce = loss_gg + loss_lg
            loss_nce /= 6
            losses_g.update(loss_gg.item() / 2, b)
            dict_result['Loss_g'].append(losses_g.output())
            losses_l.update(loss_lg.item() / 4, b)
            dict_result['Loss_l'].append(losses_l.output())
        else:
            loss_nce = nce(q[:b], k[b:])

        losses.update(loss_nce.item(), b)
        dict_result['Loss'].append(losses.output())

        optimizer.zero_grad()
        loss_nce.backward()
        norm0 = nn.utils.clip_grad_norm_(model.module.encoder_q.parameters(), 100)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    data_frame = pd.DataFrame(data=dict_result)
    data_frame.to_csv(os.path.join(args.store_path,
                                   'record', f'record_epoch_{epoch}.csv'))


#util

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def output(self):
        return self.avg
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def nce_loss(q, k, neg, T, target, criterion, get_logits=False):
    '''return logits for moco'''
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    # negative logits: NxK
    l_neg = torch.einsum('nc,ck->nk', [q, neg])
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)
    # apply temperature
    logits /= T
    if get_logits:
        return criterion(logits, target), logits
    else:
        return criterion(logits, target)