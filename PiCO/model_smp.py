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

    def predict(self, img_q, args=None, eval_only=False):
        
        _, q = self.encoder_q(img_q) # logits和归一化的

        return q

    # def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):
    def forward(self, img_q, args=None, eval_only=False):
        
        output, q = self.encoder_q(img_q) # logits和归一化的embedding
        # if eval_only:
        #     return output

        return output
