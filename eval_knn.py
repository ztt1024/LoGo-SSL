import os
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import pandas as pd

import utils
from logo.loader import load_eval_datasets
from logo.backbone import Encoder_Base


def extract_feature_pipeline(args, pretrained, method):
    # ============ preparing data ... ============
    dataset_train, dataset_val = load_eval_datasets(args.dataset, args.data_path)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = Encoder_Base(args.arch, args.dim, args.dataset, method)
    print(f"Model {args.arch}  built.")
    model.cuda()
    checkpoint = torch.load(pretrained)
    load_model(model, checkpoint, method)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val)

    train_labels = torch.tensor(dataset_train.targets).cuda().long()
    test_labels = torch.tensor(dataset_val.targets).cuda().long()
    # save features and labels
    return [train_features, test_features, train_labels, test_labels]


@torch.no_grad()
def extract_features(model, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        _,feats  = model(samples)
        feats = feats.detach()
        feats = nn.functional.normalize(feats)
        feats = feats.reshape([feats.shape[0], feats.shape[1]])
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if args.use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if args.use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def load_model(model, checkpoint, method):
    model.fc = nn.modules.Flatten(start_dim=1)
    print(model)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if method == 'simsiam':
            if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                # remove prefix
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        elif method == 'reg' or method == 'reg2':
            if k.startswith('module.online_encoder') and not k.startswith('module.online_encoder.fc'):
                # remove prefix
                state_dict[k[len("module.online_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        elif method == 'moco':
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        elif method == 'byol':
            if k.startswith('module.online_encoder') and not k.startswith('module.online_encoder.fc'):
                # remove prefix
                state_dict[k[len("module.online_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    print(model.load_state_dict(state_dict, str))


def main_worker(args, acc_list, pretrained, method):
    # need to extract features !
    output = extract_feature_pipeline(args, pretrained, method)
    torch.cuda.empty_cache()
    if dist.get_rank() == 0:
        print("Features are ready!\nStart the k-NN classification.")
        top1, top5 = knn_classifier(output[0], output[2],
                                    output[1], output[3], args.nb_knn, args.temperature)
        print(f"{args.nb_knn}-NN classifier result: Top1: {top1}, Top5: {top5}")
        acc_list[0] = top1
        acc_list[1] = top5
        print(acc_list)
    torch.cuda.empty_cache()
    dist.barrier()



def str_cat(ls, split):
    pretrained = ls[0]
    for c in ls[1:]:
        pretrained += split + c
    return pretrained


def set_pretrained(path, epoch, root):
    str_ls = path.split('/')[-1].split('_')
    str_ls[-2] = '{:04d}'.format(epoch)
    filename = str_cat(str_ls, '_')
    return os.path.join(root, filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=20, type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('--use-cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--num-workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data-path', default='/path/to/imagenet/', type=str)

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34')
    parser.add_argument('--store-path', default='', type=str, help='path to save result')
    parser.add_argument('--dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--pred-dim', default=512, type=int,
                        help='hidden dimension of the predictor (default: 512)')

    parser.add_argument('--eval-step', default=1, type=int)
    parser.add_argument('--dataset', default='', type=str, help='dataset used to pretrain')
    parser.add_argument('--end', default=None, type=int, help='end epoch for eval')

    args = parser.parse_args()

    root = args.pretrained.split('/')[:-1]
    root = str_cat(root, '/')
    str_ls = args.pretrained.split('/')[-1].split('_')
    start_epoch = epoch = int(str_ls[-2])
    method = str_ls[0]
    acc_list = [[], []]
    result = {'acc1': [], 'acc5': []}
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    print(f'loading {args.pretrained}')

    path = os.path.join(args.store_path, 'eval')
    if not os.path.exists(path) and utils.get_rank() == 0:
        os.makedirs(path)

    if args.end is not None:
        pretrained = args.pretrained
        while epoch <= args.end:
            if not os.path.isfile(pretrained):
                break
            main_worker(args, acc_list, pretrained, method)
            epoch += args.eval_step
            pretrained = set_pretrained(pretrained, epoch, root)
            if utils.get_rank() == 0:
                print(acc_list)
                acc_list = torch.tensor(acc_list)
                result['acc1'].append(float(acc_list[0]))
                result['acc5'].append(float(acc_list[1]))

    else:
        main_worker(args, acc_list, args.pretrained, method)
        if utils.get_rank() == 0:
            print(acc_list)
            acc_list = torch.tensor(acc_list)
            result['acc1'].append(float(acc_list[0]))
            result['acc5'].append(float(acc_list[1]))
    if utils.get_rank() == 0:
        data_frame = pd.DataFrame(data=result)
        data_frame.to_csv(os.path.join(args.store_path,'eval','knn_eval_{}_{}_st{}.csv'.format
        (start_epoch, args.end, args.eval_step)))


