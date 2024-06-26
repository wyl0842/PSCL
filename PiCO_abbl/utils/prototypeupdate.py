import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

# class PiCO(nn.Module):

#     def __init__(self, args, base_encoder):
#         super().__init__()
        
#         pretrained = args.dataset == 'cub200'
#         # we allow pretraining for CUB200, or the network will not converge

#         self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
#         # momentum encoder
#         self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)

#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient

#         # create the queue
#         # register_buffer定义的参数不会随着optim.step更新
#         self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))
#         self.register_buffer("queue_pseudo", torch.randn(args.moco_queue, args.num_class))
#         self.register_buffer("queue_nlabels", torch.randn(args.moco_queue, args.num_class))
#         # self.register_buffer("queue_partial", torch.randn(args.moco_queue, args.num_class))
#         self.queue = F.normalize(self.queue, dim=0)
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
#         self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))

#     @torch.no_grad()
#     def _momentum_update_key_encoder(self, args):
#         """
#         update momentum encoder
#         """
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

#     @torch.no_grad()
#     # def _dequeue_and_enqueue(self, keys, labels, partial_Y, args):
#     def _dequeue_and_enqueue(self, keys, labels, nlabels, args):
#         # gather keys before updating queue
#         keys = concat_all_gather(keys)
#         labels = concat_all_gather(labels)
#         nlabels = concat_all_gather(nlabels)
#         # partial_Y = concat_all_gather(partial_Y)

#         batch_size = keys.shape[0]

#         ptr = int(self.queue_ptr)
#         assert args.moco_queue % batch_size == 0  # for simplicity

#         # replace the keys at ptr (dequeue and enqueue)
#         # 一个队列中有batch_size个值，ptr从0开始
#         self.queue[ptr:ptr + batch_size, :] = keys
#         self.queue_pseudo[ptr:ptr + batch_size, :] = labels
#         self.queue_nlabels[ptr:ptr + batch_size, :] = nlabels
#         # self.queue_partial[ptr:ptr + batch_size, :] = partial_Y
#         ptr = (ptr + batch_size) % args.moco_queue  # move pointer

#         self.queue_ptr[0] = ptr

#     @torch.no_grad()
#     # def _batch_shuffle_ddp(self, x, y, p_y):
#     def _batch_shuffle_ddp(self, x, y):
#         """
#         Batch shuffle, for making use of BatchNorm.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         x_gather = concat_all_gather(x)
#         y_gather = concat_all_gather(y)
#         # p_y_gather = concat_all_gather(p_y)
#         batch_size_all = x_gather.shape[0]

#         num_gpus = batch_size_all // batch_size_this

#         # random shuffle index
#         idx_shuffle = torch.randperm(batch_size_all).cuda()

#         # broadcast to all gpus
#         torch.distributed.broadcast(idx_shuffle, src=0)

#         # index for restoring
#         idx_unshuffle = torch.argsort(idx_shuffle)

#         # shuffled index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

#         # return x_gather[idx_this], y_gather[idx_this], p_y_gather[idx_this], idx_unshuffle
#         return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

#     @torch.no_grad()
#     # def _batch_unshuffle_ddp(self, x, y, p_y, idx_unshuffle):
#     def _batch_unshuffle_ddp(self, x, y, idx_unshuffle):
#         """
#         Undo batch shuffle.
#         *** Only support DistributedDataParallel (DDP) model. ***
#         """
#         # gather from all gpus
#         batch_size_this = x.shape[0]
#         x_gather = concat_all_gather(x)
#         y_gather = concat_all_gather(y)
#         # p_y_gather = concat_all_gather(p_y)

#         batch_size_all = x_gather.shape[0]

#         num_gpus = batch_size_all // batch_size_this

#         # restored index for this gpu
#         gpu_idx = torch.distributed.get_rank()
#         idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

#         # return x_gather[idx_this], y_gather[idx_this], p_y_gather[idx_this]
#         return x_gather[idx_this], y_gather[idx_this]

#     def reset_prototypes(self, prototypes):
#         self.prototypes = prototypes

#     # def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):
#     def forward(self, img_q, im_k=None, nlabels=None, args=None, eval_only=False):
        
#         output, q = self.encoder_q(img_q) # logits和归一化的embedding
#         if eval_only:
#             return output
#         # for testing

#         # # 先用候选标签把确定为negative的标签筛掉
#         # predicted_scores = torch.softmax(output, dim=1) * partial_Y
#         predicted_scores = torch.softmax(output, dim=1)
#         # 得到预测的伪标签
#         max_scores, pseudo_labels = torch.max(predicted_scores, dim=1)
#         # using partial labels to filter out negative labels

#         # compute protoypical logits
#         prototypes = self.prototypes.clone().detach()
#         # 计算与原型的相似度
#         logits_prot = torch.mm(q, prototypes.t())
#         score_prot = torch.softmax(logits_prot, dim=1)

#         # update momentum prototypes with pseudo labels
#         # 原型更新
#         for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels)):
#             self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
#         # normalize prototypes    
#         self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
#         # compute key features 
#         with torch.no_grad():  # no gradient 
#             self._momentum_update_key_encoder(args)  # update the momentum encoder
#             # shuffle for making use of BN
#             # im_k, predicted_scores, partial_Y, idx_unshuffle = self._batch_shuffle_ddp(im_k, predicted_scores, partial_Y)
#             im_k, predicted_scores, idx_unshuffle = self._batch_shuffle_ddp(im_k, predicted_scores)
#             _, k = self.encoder_k(im_k)
#             # undo shuffle
#             # k, predicted_scores, partial_Y = self._batch_unshuffle_ddp(k, predicted_scores, partial_Y, idx_unshuffle)
#             k, predicted_scores = self._batch_unshuffle_ddp(k, predicted_scores, idx_unshuffle)

#         features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
#         pseudo_scores = torch.cat((predicted_scores, predicted_scores, self.queue_pseudo.clone().detach()), dim=0)
#         noisy_labels = torch.cat((nlabels, nlabels, self.queue_nlabels.clone().detach()), dim=0)
#         # partial_target = torch.cat((partial_Y, partial_Y, self.queue_partial.clone().detach()), dim=0)
#         # to calculate SupCon Loss using pseudo_labels and partial target
        
#         # dequeue and enqueue
#         # self._dequeue_and_enqueue(k, predicted_scores, partial_Y, args)
#         self._dequeue_and_enqueue(k, predicted_scores, nlabels, args)

#         # return output, features, pseudo_scores, partial_target, score_prot
#         # print(pseudo_scores.shape)
#         # print(noisy_labels.shape)
#         return output, features, pseudo_scores, noisy_labels, score_prot

# # utils
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output


# class partial_loss(nn.Module):
#     def __init__(self, confidence, conf_ema_m=0.99):
#         super().__init__()
#         self.confidence = confidence
#         # self.init_conf = confidence.detach()
#         self.conf_ema_m = conf_ema_m

#     def set_conf_ema_m(self, epoch, args):
#         start = args.conf_ema_range[0]
#         end = args.conf_ema_range[1]
#         self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

#     # outputs是模型预测，confidence是根据原型和候选标签得到的伪标签
#     def forward(self, outputs, index):
#         logsm_outputs = F.log_softmax(outputs, dim=1)
#         final_outputs = logsm_outputs * self.confidence[index, :]
#         average_loss = - ((final_outputs).sum(dim=1)).mean()
#         return average_loss
    
#     def confidence_update(self, temp_un_conf, batch_index, batchY):
#         with torch.no_grad():
#             # _, prot_pred = (temp_un_conf * batchY).max(dim=1) # 用候选标签把确定为negative的标签设置为0
#             _, prot_pred = temp_un_conf.max(dim=1) # 用相似度确定校正后的标签
#             pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
#             self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :]\
#                  + (1 - self.conf_ema_m) * pseudo_label
#         return None

class prototype(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.register_buffer("prototypes", torch.zeros(3*args.num_class,args.low_dim))
    
    def reset_prototypes(self, prototypes):
        self.prototypes = prototypes

    def forward(self, feat, label):
        # update momentum prototypes
        # 原型更新
        self.prototypes[3*label:3*label+2] = feat
        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        return self.prototypes