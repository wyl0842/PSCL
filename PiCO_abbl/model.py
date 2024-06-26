import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class PiCO(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        
        pretrained = args.dataset == 'cub200'
        # we allow pretraining for CUB200, or the network will not converge

        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
      
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))

    def reset_prototypes(self, prototypes):
        self.prototypes = prototypes

    # def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):
    def forward(self, img_q, args=None, eval_only=False):
        
        output, q = self.encoder_q(img_q) # logits和归一化的embedding
        if eval_only:
            return output
        # for testing

        # # 先用候选标签把确定为negative的标签筛掉
        # predicted_scores = torch.softmax(output, dim=1) * partial_Y
        predicted_scores = torch.softmax(output, dim=1)
        # 得到预测的伪标签
        max_scores, pseudo_labels = torch.max(predicted_scores, dim=1)
        # using partial labels to filter out negative labels

        # compute protoypical logits
        prototypes = self.prototypes.clone().detach()
        # 计算与原型的相似度
        logits_prot = torch.mm(q, prototypes.t())
        with torch.no_grad():
            score_prot = torch.softmax(logits_prot, dim=1)

        # update momentum prototypes with pseudo labels
        # 原型更新
        for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels)):
            self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        return output, score_prot

# utils
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
