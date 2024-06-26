# CUDA_VISIBLE_DEVICES=0,1 python representation_plot_cifar.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
# 需要同步修改 from model import PiCO; args.resume; 保存图片名

"""
Date: 28/07/2017
feature exploration and visualization

Author: Xingjun Ma
"""
import os
import argparse
import numpy as np
import torch
from resnet import *
from model import PiCO
# from model_abbl import PiCO
# from model_smp import PiCO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import data_load_base_tsne as data_load
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from utils.utils_algo import *

matplotlib.rcParams.update({'font.size': 32})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['savefig.dpi'] = 300 #图片像素
np.random.seed(1234)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2022 Oral paper PiCO')
parser.add_argument('--dataset', default='cifar10', type=str, 
                    choices=['cifar10', 'cifar100', 'cub200'],
                    help='dataset name (cifar10)')
parser.add_argument('--exp-dir', default='experiment/PiCO', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture (only resnet18 used in PiCO)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int, 
                    help='queue size; number of negative samples')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

args = parser.parse_args()

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target

train_dataset = data_load.cifar10_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=transform_target,
                                        dataset='cifar10',
                                        noise_type='symmetric',
                                        noise_rate=0.4,
                                        split_per=1,
                                        random_seed=1)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=128,
                                            num_workers=4,
                                            shuffle=False)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
clf1 = PiCO(args, SupConResNet)
# print(clf1)

# pico
# args.resume = '/home/wangyl/Code/PSCL/PiCO/experiment/PSCL1116-CIFAR-10/ds_cifar10_nr_0.4_nt_symmetric_lr_0.01_ep_800_ps_80_lw_0.5_pm_0.99_arch_resnet18_heir_False_sd_1/checkpoint.pth.tar'
# pico_cl
# args.resume = '/home/wangyl/Code/PSCL/PiCO_CL/experiment/PSCL231015dist-CIFAR-10-MOCO/ds_cifar10_nr_0.4_nt_symmetric_lr_0.01_ep_800_ps_80_lw_0.1_pm_0.99_arch_resnet18_heir_False_sd_1/checkpoint_79.pth.tar'
args.resume = '/home/wangyl/Code/PSCL/PiCO_CL/experiment/PSCL231016dist-CIFAR-10/ds_cifar10_nr_0.4_nt_symmetric_lr_0.01_ep_800_ps_80_lw_0.1_pm_0.99_arch_resnet18_heir_False_sd_1/checkpoint_79.pth.tar'
# abbl 监督对比学习替换成普通监督学习
# args.resume = '/home/wangyl/Code/PSCL/PiCO/experiment/PSCLabbl-CIFAR-10-resnet/ds_cifar10_nr_0.4_nt_symmetric_lr_0.01_ep_800_ps_80_lw_0.5_pm_0.99_arch_resnet18_heir_False_sd_1/checkpoint.pth.tar'
# abbl smp
# args.resume = '/home/wangyl/Code/PSCL/PiCO_abbl_cos/experiment/PSCL230305dist-CIFAR-10-abblsmp/ds_cifar10_nr_0.4_nt_symmetric_lr_0.01_ep_100_ps_40_lw_0.5_pm_0.99_arch_resnet18_heir_False_sd_1/checkpoint_best.pth.tar'

loc = 'cuda:0'

# 用于单卡训练的模型
# loaded_dict = torch.load(args.resume)
# clf1 = nn.DataParallel(clf1).cuda()
# clf1.load_state_dict(loaded_dict['state_dict'])

# 用于并行训练的模型
checkpoint = torch.load(args.resume, map_location=loc)
clf1 = nn.DataParallel(clf1).cuda()
clf1.load_state_dict(checkpoint['state_dict'])

feat_result_output_ce = []
def get_features_hook(module, data_input, data_output):
        # feat_result_input.append(data_input)
        feat_result_output_ce.append(data_output)

def feature_visualization(args):
    """
    This is to show how features of incorretly labeled images are overffited to the wrong class.
    plot t-SNE 2D-projected deep features (right before logits).
    This will generate 3 plots in a grid (3x1). 
    The first shows the raw features projections of two classes of images (clean label + noisy label)
    The second shows the deep features learned by cross-entropy after training.
    The third shows the deep features learned using a new loss after training.
    
    :param model_name: a new model other than crossentropy(ce), can be: boot_hard, boot_soft, forward, backward, lid
    :param dataset: 
    :param num_classes:
    :param noise_type;
    :param noise_ratio: 
    :param epochs: to find the last epoch
    :param n_samples: 
    :return: 
    """

    clean_targets_list = []
    noisy_targets_list = []
    outputs_list = []
    
    with torch.no_grad():
        handle = clf1.module.encoder_q.encoder.avgpool.register_forward_hook(get_features_hook)
        for idx, (inputs, targets, clean_targets, _) in enumerate(train_loader):
            feat_result_output_ce.clear()
            inputs = inputs.cuda()
            targets = targets.cuda()
            clean_targets = clean_targets.cuda()
            targets_np = targets.data.cpu().numpy()
            clean_targets_np = clean_targets.data.cpu().numpy()

            outputs = clf1(inputs, args, eval_only=True)
            feat_ce = feat_result_output_ce[0]
            feat_out_ce = feat_ce.view(feat_ce.size(0), -1)
            feat_out_ce_np = feat_out_ce.data.cpu().numpy()
            # outputs_np = outputs.data.cpu().numpy()
            
            noisy_targets_list.append(targets_np[:, np.newaxis])
            clean_targets_list.append(clean_targets_np[:, np.newaxis])
            outputs_list.append(feat_out_ce_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(train_loader)):
                print(idx+1, '/', len(train_loader))
            if idx > 20:
                break

    cleantargets = np.concatenate(clean_targets_list, axis=0)
    noisytargets = np.concatenate(noisy_targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return cleantargets, noisytargets, outputs

def tsne_plot(cleantargets, noisytargets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    # tsne = TSNE(random_state=0)
    tsne = TSNE()
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['True'] = cleantargets
    df['Noisy'] = noisytargets

    # sns.set(font_scale = 2)

    # plt.rcParams['figure.figsize'] = 10, 10
    # ax = sns.scatterplot(
    #     x='x', y='y',
    #     hue='Noisy',
    #     palette=sns.color_palette("hls", 10),
    #     data=df,
    #     marker='o',
    #     legend="full",
    #     alpha=0.5
    # )
 
    # legend = ax.legend(loc=0, labels=CLASSES, handletextpad=0.2, labelspacing=0, borderpad=0.1, handlelength=0.3)
    # plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
    # plt.setp(ax.get_legend().get_title(), fontsize='28') # for legend title
    # # ax.legend(bbox_to_anchor=(0.5, -0.2), columnspacing=0.2, loc=8, ncol=10, borderpad=0.2, labelspacing=0.2, handlelength=1, handletextpad=0, title='Noisy Label')

    # plt.xticks([])
    # plt.yticks([])
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.title('Noisy Label', fontsize = 24)

    # plt.savefig('tsne_abbl_230330_Noisy.png', bbox_inches='tight')
    # # plt.savefig('tsne_clean_ce_0913.png', bbox_inches='tight')


    plt.rcParams['figure.figsize'] = 10, 10
    ax = sns.scatterplot(
        x='x', y='y',
        hue='True',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    legend = ax.legend(loc=0, labels=CLASSES, handletextpad=0.2, labelspacing=0, borderpad=0.1, handlelength=0.3)
    plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='24') # for legend title
    # ax.legend(bbox_to_anchor=(0.5, -0.2), columnspacing=0.2, loc=8, ncol=10, borderpad=0.2, labelspacing=0.2, handlelength=1, handletextpad=0, title='True Label')

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('True Label', fontsize = 24)
    plt.savefig('tsne_80_231028_True.png', bbox_inches='tight')
    # plt.savefig('tsne_noisy_ce_0913.png', bbox_inches='tight')

    print('done!')

def test(model, test_loader, args, epoch, tb_logger):
    with torch.no_grad():
        print('==> Evaluation...')       
        model.eval()    
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        for batch_idx, (images, _, labels, _) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images, args, eval_only=True)    
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])
        
        # average across all processes
        # acc_tensors = torch.Tensor([top1_acc.avg,top5_acc.avg]).cuda(args.gpu)
        acc_tensors = torch.Tensor([top1_acc.avg,top5_acc.avg]).cuda()
        # dist.all_reduce(acc_tensors)        
        # acc_tensors /= args.world_size
        
        print('Accuracy is %.2f%% (%.2f%%)'%(acc_tensors[0],acc_tensors[1]))
        if args.gpu ==0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('Top5 Acc', acc_tensors[1], epoch)             
    return acc_tensors[0]

if __name__ == "__main__":
    # acc_test = test(clf1, train_loader, args, 800, None)
    cleantargets, noisytargets, outputs = feature_visualization(args)
    tsne_plot(cleantargets, noisytargets, outputs)