import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision.datasets import ImageFolder
from .randaugment import RandomAugment
from utils import tools

# def load_cifar10(partial_rate, batch_size):
#     test_transform = transforms.Compose(
#             [transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
#     temp_train = dsets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
#     data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()
#     # get original data and labels

#     test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
#         sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
#     # set test dataloader
    
#     # 产生的候选集实际是随机选择若干个类别标为1，其他为0
#     partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
#     # generate partial labels
#     temp = torch.zeros(partialY.shape)
#     temp[torch.arange(partialY.shape[0]), labels] = 1
#     if torch.sum(partialY * temp) == partialY.shape[0]:
#         print('partialY correctly loaded')
#     else:
#         print('inconsistent permutation')

#     print('Average candidate num: ', partialY.sum(1).mean())
#     partial_matrix_dataset = CIFAR10_Augmentention(data, partialY.float(), labels.float())
#     # generate partial label dataset

#     train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
#     partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
#         batch_size=batch_size, 
#         shuffle=(train_sampler is None), 
#         num_workers=4,
#         pin_memory=True,
#         sampler=train_sampler,
#         drop_last=True)
#     return partial_matrix_train_loader,partialY,train_sampler,test_loader


# class CIFAR10_Augmentention(Dataset):
#     def __init__(self, images, given_label_matrix, true_labels):
#         self.images = images
#         self.given_label_matrix = given_label_matrix
#         # user-defined label (partial labels)
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
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
#         self.strong_transform = transforms.Compose(
#             [
#             transforms.ToPILImage(),
#             transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#             transforms.RandomHorizontalFlip(),
#             RandomAugment(3, 5),
#             transforms.ToTensor(), 
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

#     def __len__(self):
#         return len(self.true_labels)
        
#     def __getitem__(self, index):
#         each_image_w = self.weak_transform(self.images[index])
#         each_image_s = self.strong_transform(self.images[index])
#         each_label = self.given_label_matrix[index]
#         each_true_label = self.true_labels[index]
        
#         return each_image_w, each_image_s, each_label, each_true_label, index


def load_tire(batch_size, noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):

    noisy_dataset = train_dataset(root='./data/train', train=True, noise_rate=noise_rate, split_per=split_per, random_seed=random_seed, num_class=num_class)
    train_labels = noisy_dataset.train_labels
    train_paths = noisy_dataset.paths
    train_sampler = torch.utils.data.distributed.DistributedSampler(noisy_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=noisy_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    # test_loader
    # test_data = np.load('data/cifar10/test_images.npy')
    # test_labels = np.load('data/cifar10/test_labels.npy')
    te_dataset = test_dataset(root='./data/test')

    test_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
        sampler=torch.utils.data.distributed.DistributedSampler(te_dataset, shuffle=False))
    # set test dataloader
    return train_loader, train_labels, train_sampler, test_loader, train_paths


# class CIFAR10_Augmentention(Dataset):
#     def __init__(self, images, noisy_labels, true_labels):
#         self.images = images
#         self.noisy_labels = noisy_labels
#         # user-defined label (partial labels)
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
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
#         self.strong_transform = transforms.Compose(
#             [
#             transforms.ToPILImage(),
#             transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
#             transforms.RandomHorizontalFlip(),
#             RandomAugment(3, 5),
#             transforms.ToTensor(), 
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

#     def __len__(self):
#         return len(self.true_labels)
        
#     def __getitem__(self, index):
#         each_image_w = self.weak_transform(self.images[index])
#         each_image_s = self.strong_transform(self.images[index])
#         each_noisy_label = self.noisy_labels[index]
#         each_true_label = self.true_labels[index]
        
#         return each_image_w, each_image_s, each_noisy_label, each_true_label, index

# class CIFAR10_Augmentention_test(Dataset):
#     def __init__(self, images, labels):
#         self.images = images
#         self.labels = labels
#         self.test_transform = transforms.Compose(
#             [transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

#     def __len__(self):
#         return len(self.labels)
        
#     def __getitem__(self, index):
#         each_image = self.test_transform(self.images[index])
#         each_label = self.labels[index]
        
#         return each_image, each_label


class train_dataset(ImageFolder):
    def __init__(self, root, train=True, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=100):            
        super(train_dataset, self).__init__(root)
        
        # self.indices = range(len(self)) #该文件夹中的长度
        # self.weak_transform = transforms.Compose(
        #     [
        #     transforms.RandomResizedCrop(size=256, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(), 
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # self.strong_transform = transforms.Compose(
        #     [
        #     transforms.RandomResizedCrop(size=256, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(),
        #     RandomAugment(3, 5),
        #     transforms.ToTensor(), 
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.weak_transform = transforms.Compose(
            [
            transforms.RandomResizedCrop(size=256, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.Grayscale(),
            transforms.ToTensor(), 
            transforms.Normalize([0.5,], [0.5,])])
        self.strong_transform = transforms.Compose(
            [
            transforms.RandomResizedCrop(size=256, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Grayscale(),
            transforms.Normalize([0.5,], [0.5,])])
        self.noise_rate = noise_rate
        self.num_class = num_class
        self.random_seed = random_seed
        self.split_per = split_per
        self.train = train
        self.path = [item[0] for item in self.imgs]
        self.label = [item[1] for item in self.imgs]
        self.img = [self.loader(path_item) for path_item in self.path]

        # 为标签加噪声
        self.noisy_labels = np.array(self.label)
        probs_to_change = torch.randint(100, (len(self.noisy_labels),))
        idx_to_change = probs_to_change >= (100.0 - 100 * self.noise_rate)
        for n, _ in enumerate(self.noisy_labels):
            if idx_to_change[n] == 1:
                set_labels = list(set(range(self.num_class)))  # this is a set with the available labels (with the current label)
                set_index = np.random.randint(len(set_labels))
                self.noisy_labels[n] = set_labels[set_index]
        
        # 划分数据集9：1
        num_samples = len(self.noisy_labels)
        np.random.seed(self.random_seed)
        train_set_index = np.random.choice(num_samples, int(num_samples*self.split_per), replace=False)
        all_index = np.arange(num_samples)
        val_set_index = np.delete(all_index, train_set_index)
        self.noisy_labels = np.eye(self.num_class, dtype=np.float)[self.noisy_labels]

        self.train_data, self.train_labels, self.true_labels, self.val_data, self.val_labels = [], [], [], [], []
        self.paths = [] # 用于记录校正过程
        for idx in train_set_index:
            self.train_data.append(self.img[idx])
            self.train_labels.append(self.noisy_labels[idx])
            self.true_labels.append(self.label[idx])
            self.paths.append(self.path[idx]) # 用于记录校正过程

        for idx in val_set_index:
            self.val_data.append(self.img[idx])
            self.val_labels.append(self.noisy_labels[idx])

    def __getitem__(self, index):
        
        if self.train:
            img, label, true_label = self.train_data[index], self.train_labels[index], self.true_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]
        
        # if self.transform is not None:
        #     img = self.transform(img)

        # return img, label, index
        each_image_w = self.weak_transform(img)
        each_image_s = self.strong_transform(img)
        each_noisy_label = label
        each_true_label = true_label
        
        return each_image_w, each_image_s, each_noisy_label, each_true_label, index
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)
        # return len(self.imgs)

class test_dataset(ImageFolder):
    def __init__(self, root):
            
        super(test_dataset, self).__init__(root)
        self.indices = range(len(self)) #该文件夹中的长度
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize([0.5,], [0.5,])])

    def __getitem__(self, index):
           
        path = self.imgs[index][0] #此时的imgs等于samples，即内容为[(图像路径, 该图像对应的类别索引值),(),...]
        label = self.imgs[index][1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
     
        return img, label
    def __len__(self):
        return len(self.imgs)