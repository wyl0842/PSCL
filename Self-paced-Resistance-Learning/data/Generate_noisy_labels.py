import os
import os.path
import copy
import hashlib
import errno
import numpy as np
from numpy.testing import assert_array_almost_equal

# def check_integrity(fpath, md5):
#     if not os.path.isfile(fpath):
#         return False
#     md5o = hashlib.md5()
#     with open(fpath, 'rb') as f:
#         # read in 1MB chunks
#         for chunk in iter(lambda: f.read(1024 * 1024), b''):
#             md5o.update(chunk)
#     md5c = md5o.hexdigest()
#     if md5c != md5:
#         return False
#     return True


# def download_url(url, root, filename, md5):
#     from six.moves import urllib

#     root = os.path.expanduser(root)
#     fpath = os.path.join(root, filename)

#     try:
#         os.makedirs(root)
#     except OSError as e:
#         if e.errno == errno.EEXIST:
#             pass
#         else:
#             raise

#     # downloads file
#     if os.path.isfile(fpath) and check_integrity(fpath, md5):
#         print('Using downloaded and verified file: ' + fpath)
#     else:
#         try:
#             print('Downloading ' + url + ' to ' + fpath)
#             urllib.request.urlretrieve(url, fpath)
#         except:
#             if url[:5] == 'https':
#                 url = url.replace('https:', 'http:')
#                 print('Failed download. Trying https -> http instead.'
#                       ' Downloading ' + url + ' to ' + fpath)
#                 urllib.request.urlretrieve(url, fpath)


# def list_dir(root, prefix=False):
#     """List all directories at a given root

#     Args:
#         root (str): Path to directory whose folders need to be listed
#         prefix (bool, optional): If true, prepends the path to each result, otherwise
#             only returns the name of the directories found
#     """
#     root = os.path.expanduser(root)
#     directories = list(
#         filter(
#             lambda p: os.path.isdir(os.path.join(root, p)),
#             os.listdir(root)
#         )
#     )

#     if prefix is True:
#         directories = [os.path.join(root, d) for d in directories]

#     return directories


# def list_files(root, suffix, prefix=False):
#     """List all files ending with a suffix at a given root

#     Args:
#         root (str): Path to directory whose folders need to be listed
#         suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
#             It uses the Python "str.endswith" method and is passed directly
#         prefix (bool, optional): If true, prepends the path to each result, otherwise
#             only returns the name of the files found
#     """
#     root = os.path.expanduser(root)
#     files = list(
#         filter(
#             lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
#             os.listdir(root)
#         )
#     )

#     if prefix is True:
#         files = [os.path.join(root, d) for d in files]

#     return files

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print (np.max(y))
    print(P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    print (m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1

        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise
    actual_noise=0.0
    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy
    # print (P)
    print('Actual noise %.2f' % actual_noise)
    return y_train, actual_noise

def noisify_multiclass_asymmetric_cifar10(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    source_class = [9, 2, 3, 5, 4]
    target_class = [1, 0, 5, 3, 7]
    y_train_ = y_train
    for s, t in zip(source_class, target_class):
        cls_idx = np.where(np.array(y_train) == s)[0]
        n_noisy = int(noise * cls_idx.shape[0])
        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
        for idx in noisy_sample_index:
            y_train_[idx] = t
    return y_train_, source_class, target_class

def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def noisify_multiclass_asymmetric_cifar100(y_train, noise, random_state=None, nb_classes=100):
    """mistakes:
        flip in the symmetric way
    """
    nb_classes = 100
    P = np.eye(nb_classes)
    n = noise
    nb_superclasses = 20
    nb_subclasses = 5

    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i + 1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

            y_train_noisy = multiclass_noisify(np.array(y_train), P=P, random_state=random_state)
            actual_noise = (y_train_noisy != np.array(y_train)).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        targets = y_train_noisy
    return targets, actual_noise, P

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P
    actual_noise=0.0
    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy
    # print (P)
    print('Actual noise %.2f' % actual_noise)
    return y_train, actual_noise

def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate

def noisify_asym(nb_classes=10, train_labels=None, dataset='cifar10', noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'asymmetric' and dataset == 'cifar10':
        train_noisy_labels, _, _ = noisify_multiclass_asymmetric_cifar10(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'asymmetric' and dataset == 'cifar100':
        train_noisy_labels, _, _ = noisify_multiclass_asymmetric_cifar100(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels