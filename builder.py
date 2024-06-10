import torch
import torch.nn as nn
from logo.backbone import Encoder_Base
from torch.nn.functional import softplus
import torch.nn.functional as F

class MoCo_LoGo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, arch, dim=128, K=65536, m=0.999, T=0.07, dataset='imagenet100'):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_LoGo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = Encoder_Base(arch, dim, dataset, 'moco')
        self.encoder_k = Encoder_Base(arch, dim, dataset, 'moco')

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, img, local = False):
        """
        Input:
            img: crops from a batch of instances
        Output:
            q: query feature
            k: key feature
            neg: feature from negative instances
        """
        # compute query features
        b = img[0].shape[0]
        for idx, im in enumerate(img):
            _, _q = self.encoder_q(im)
            if idx == 0:
                q = _q
            else:
                q = torch.cat([q, _q])
        q = nn.functional.normalize(q)

        # compute key features

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        if local is False:
            with torch.no_grad():
            # update the key encoder
            # shuffle for making use of BN
                for idx, im in enumerate(img):
                    im, idx_unshuffle = self._batch_shuffle_ddp(im)

                    _, _k = self.encoder_k(im)  # keys: NxC
                    _k = nn.functional.normalize(_k, dim=1)

                    # undo shuffle
                    _k = self._batch_unshuffle_ddp(_k, idx_unshuffle)
                    if idx == 0:
                        k = _k
                    else:
                        k = torch.cat([k, _k])

            neg = self.queue.clone().detach()

            self._dequeue_and_enqueue(k[:2 * b])

            return q, k, neg
        else:
            # with torch.no_grad():
            #     self._momentum_update_key_encoder()
            return q


class Regressor_cos(nn.Module):
    """
    Build a MLP kernel for LoGo Regressor
    """
    def __init__(self, fea_dim=128, mlp_dim=256):
        """
        fea_dim: feature dim after projection head (default:128)
        mlp_dim: dim of hidden layers of Regressor
        """
        super(Regressor_cos, self).__init__()
        self.input_dim = fea_dim*2
        self.fea_dim = fea_dim
        self.mlp_dim = mlp_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mlp_dim, out_features=self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            # nn.Linear(in_features=self.mlp_dim, out_features=self.mlp_dim),
            # nn.BatchNorm1d(self.mlp_dim),
            # nn.ReLU(),
            # nn.Linear(in_features=self.mlp_dim, out_features=self.mlp_dim),
            # nn.BatchNorm1d(self.mlp_dim),
            # nn.ReLU(),
            # nn.Linear(in_features=self.mlp_dim, out_features=1)
            nn.Linear(in_features=self.mlp_dim, out_features=self.fea_dim)
        )

    def forward(self, x, input_type = 'neg'):
        output = self.mlp(x)
        if input_type == 'neg':
            # When local crops from different instances
            # output = softplus(output)
            # z1, z2 = output[:, :self.fea_dim], output[:, self.fea_dim:]
            # output = nn.functional.cosine_similarity(z1, z2)
            z1 = x[:, self.fea_dim:]
            z2 = output
            #z2 = nn.functional.normalize(z2)
            output = nn.functional.cosine_similarity(z1, z2)
        elif input_type == 'pos':
            # When local crops from same instance
            # output = softplus(-output)
            # z1, z2 = output[:, :self.fea_dim], output[:, self.fea_dim:]
            # output = -nn.functional.cosine_similarity(z1, z2)
            z1 = x[:, self.fea_dim:]
            z2 = output
            #z2 = nn.functional.normalize(z2)
            output = -nn.functional.cosine_similarity(z1, z2)
        return output


class Regressor_softplus(nn.Module):
    """
    Build a MLP kernel for LoGo Regressor
    """
    def __init__(self, fea_dim=128, mlp_dim=256):
        """
        fea_dim: feature dim after projection head (default:128)
        mlp_dim: dim of hidden layers of Regressor
        """
        super(Regressor_softplus, self).__init__()
        self.input_dim = fea_dim*2
        self.fea_dim = fea_dim
        self.mlp_dim = mlp_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mlp_dim, out_features=self.mlp_dim),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(),
            # nn.Linear(in_features=self.mlp_dim, out_features=self.mlp_dim),
            # nn.BatchNorm1d(self.mlp_dim),
            # nn.ReLU(),
            # nn.Linear(in_features=self.mlp_dim, out_features=self.mlp_dim),
            # nn.BatchNorm1d(self.mlp_dim),
            # nn.ReLU(),
            # nn.Linear(in_features=self.mlp_dim, out_features=1)
            nn.Linear(in_features=self.mlp_dim, out_features=1)
        )

    def forward(self, x, input_type = 'neg'):
        output = self.mlp(x)
        if input_type == 'neg':
            # When local crops from different instances
            output = softplus(output)
        elif input_type == 'pos':
            # When local crops from same instance
            output = softplus(-output)
        return output


#utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
