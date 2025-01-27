import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .randaugment import RandomAugment
from .utils_algo import generate_uniform_cv_candidate_labels,generate_hierarchical_cv_candidate_labels
from utils import tools

# def load_cifar100(partial_rate, batch_size, hierarchical):
#     test_transform = transforms.Compose(
#             [transforms.ToTensor(),
#             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

#     temp_train = dsets.CIFAR100(root='./data', train=True, download=True)
#     data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()
#     # get original data and labels

#     test_dataset = dsets.CIFAR100(root='./data', train=False, transform=test_transform)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
#         sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    
#     if hierarchical:
#         partialY = generate_hierarchical_cv_candidate_labels('cifar100',labels, partial_rate)
#         # for fine-grained classification
#     else:
#         partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
    
#     temp = torch.zeros(partialY.shape)
#     temp[torch.arange(partialY.shape[0]), labels] = 1
#     if torch.sum(partialY * temp) == partialY.shape[0]:
#         print('partialY correctly loaded')
#     else:
#         print('inconsistent permutation')
#     print('Average candidate num: ', partialY.sum(1).mean())
#     partial_matrix_dataset = CIFAR100_Augmentention(data, partialY.float(), labels.float())
#     train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
#     partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
#         batch_size=batch_size, 
#         shuffle=(train_sampler is None), 
#         num_workers=4,
#         pin_memory=True,
#         sampler=train_sampler,
#         drop_last=True)
#     return partial_matrix_train_loader,partialY,train_sampler,test_loader


# class CIFAR100_Augmentention(Dataset):
#     def __init__(self, images, given_label_matrix, true_labels):
#         self.images = images
#         self.given_label_matrix = given_label_matrix
#         self.true_labels = true_labels
#         self.weak_transform = transforms.Compose(
#             [
#             transforms.ToPILImage(),
#             transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.ToTensor(), 
#             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
#         self.strong_transform = transforms.Compose(
#             [
#             transforms.ToPILImage(),
#             transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#             transforms.RandomHorizontalFlip(),
#             RandomAugment(3, 5),
#             transforms.ToTensor(), 
#             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

#     def __len__(self):
#         return len(self.true_labels)
        
#     def __getitem__(self, index):
#         each_image_w = self.weak_transform(self.images[index])
#         each_image_s = self.strong_transform(self.images[index])
#         each_label = self.given_label_matrix[index]
#         each_true_label = self.true_labels[index]
#         return each_image_w, each_image_s, each_label, each_true_label, index

def load_cifar100(batch_size, noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):
    # temp_train = dsets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    # data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()
    # get original data and labels

    # train_loader
    original_images = np.load('data/cifar100/train_images.npy')
    original_labels = np.load('data/cifar100/train_labels.npy')
    
    # clean images and noisy labels (training and validation)
    train_data, val_data, train_labels, val_labels = tools.dataset_split(original_images, 
                                                                             original_labels, 'cifar100', noise_type, noise_rate, split_per, random_seed, num_class)

    train_labels = np.eye(num_class)[train_labels]
    train_data = train_data.reshape((-1, 3, 32, 32))
    train_data = train_data.transpose((0, 2, 3, 1))
    val_data = val_data.reshape((-1, 3, 32, 32))
    val_data = val_data.transpose((0, 2, 3, 1))
    noisy_dataset = CIFAR100_Augmentention(train_data, train_labels, original_labels)
    train_sampler = torch.utils.data.distributed.DistributedSampler(noisy_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=noisy_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    # test_loader
    test_data = np.load('data/cifar100/test_images.npy')
    test_labels = np.load('data/cifar100/test_labels.npy')
    test_data = test_data.reshape((-1, 3, 32, 32))
    test_data = test_data.transpose((0, 2, 3, 1)) 
    test_dataset = CIFAR100_Augmentention_test(test_data, test_labels)
    # test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
        sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    # set test dataloader
    return train_loader, train_labels, train_sampler, test_loader


class CIFAR100_Augmentention(Dataset):
    def __init__(self, images, noisy_labels, true_labels):
        self.images = images
        self.noisy_labels = noisy_labels
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_noisy_label = self.noisy_labels[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_noisy_label, each_true_label, index

class CIFAR100_Augmentention_test(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        each_image = self.test_transform(self.images[index])
        each_label = self.labels[index]
        
        return each_image, each_label