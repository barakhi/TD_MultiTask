#Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import scipy.misc as m


class MNIST_ETASK9(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, multi=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.multi = multi

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if not self._check_multi_exists():
            raise RuntimeError('Multi Task extension not found.' +
                               ' You can use download=True to download it')


        if multi:
            if self.train:
                self.train_data, \
                self.train_labels_1, self.train_labels_2, self.train_labels_3, \
                self.train_labels_4, self.train_labels_5, self.train_labels_6, \
                self.train_labels_7, self.train_labels_8, self.train_labels_9, \
                    = torch.load(os.path.join(self.root, self.processed_folder, self.multi_training_file))
            else:
                self.test_data, self.test_labels_1, self.test_labels_2, self.test_labels_3, \
                self.test_labels_4, self.test_labels_5, self.test_labels_6, \
                self.test_labels_7, self.test_labels_8, self.test_labels_9, \
                    = torch.load(os.path.join(self.root, self.processed_folder, self.multi_test_file))
        else:
            if self.train:
                self.train_data, self.train_labels = torch.load(
                    os.path.join(self.root, self.processed_folder, self.training_file))
            else:
                self.test_data, self.test_labels = torch.load(
                    os.path.join(self.root, self.processed_folder, self.test_file))


    def __getitem__(self, index):
        import matplotlib.pyplot as plt
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.multi:
            if self.train:
                img, target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9 = \
                    self.train_data[index], \
                    self.train_labels_1[index], self.train_labels_2[index], self.train_labels_3[index], \
                    self.train_labels_4[index], self.train_labels_5[index], self.train_labels_6[index], \
                    self.train_labels_7[index], self.train_labels_8[index], self.train_labels_9[index]
            else:
                img, target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9  = \
                    self.test_data[index], \
                    self.test_labels_1[index], self.test_labels_2[index], self.test_labels_3[index], \
                    self.test_labels_4[index], self.test_labels_5[index], self.test_labels_6[index], \
                    self.test_labels_7[index], self.test_labels_8[index], self.test_labels_9[index]
        else:
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.multi:
            return img, target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9
        else:
            return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))
    
    def _check_multi_exists(self):
        return  os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists() and self._check_multi_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        mnist_ims, multi_mnist_ims, extension2, extension3, extension4, extension5, extension6, \
            extension7, extension8, extension9 = read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'))
        mnist_labels, multi_mnist_labels_1, multi_mnist_labels_2, multi_mnist_labels_3, \
            multi_mnist_labels_4, multi_mnist_labels_5, multi_mnist_labels_6, \
            multi_mnist_labels_7, multi_mnist_labels_8, multi_mnist_labels_9 = \
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'), extension2, extension3, extension4, \
                            extension5, extension6, extension7, extension8, extension9)

        tmnist_ims, tmulti_mnist_ims, textension2, textension3, textension4, textension5, textension6, \
            textension7, textension8, textension9 = read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'))
        tmnist_labels, tmulti_mnist_labels_1, tmulti_mnist_labels_2, tmulti_mnist_labels_3, \
            tmulti_mnist_labels_4, tmulti_mnist_labels_5, tmulti_mnist_labels_6, \
            tmulti_mnist_labels_7, tmulti_mnist_labels_8, tmulti_mnist_labels_9 = \
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'), textension2, textension3, textension4,
                    textension5, textension6, textension7, textension8, textension9)


        mnist_training_set = (mnist_ims, mnist_labels)
        multi_mnist_training_set = (multi_mnist_ims, multi_mnist_labels_1, multi_mnist_labels_2, multi_mnist_labels_3, \
            multi_mnist_labels_4, multi_mnist_labels_5, multi_mnist_labels_6, \
            multi_mnist_labels_7, multi_mnist_labels_8, multi_mnist_labels_9)

        mnist_test_set = (tmnist_ims, tmnist_labels)
        multi_mnist_test_set = (tmulti_mnist_ims, tmulti_mnist_labels_1, tmulti_mnist_labels_2, tmulti_mnist_labels_3, \
            tmulti_mnist_labels_4, tmulti_mnist_labels_5, tmulti_mnist_labels_6, \
            tmulti_mnist_labels_7, tmulti_mnist_labels_8, tmulti_mnist_labels_9)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(mnist_test_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_training_file), 'wb') as f:
            torch.save(multi_mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_test_file), 'wb') as f:
            torch.save(multi_mnist_test_set, f)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path, extension, extension2, extension3, extension4, extension5, extension6, extension7, extension8):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        multi_labels_1 = np.zeros((1*length),dtype=np.long)
        multi_labels_2 = np.zeros((1*length),dtype=np.long)
        multi_labels_3 = np.zeros((1 * length), dtype=np.long)
        multi_labels_4 = np.zeros((1 * length), dtype=np.long)
        multi_labels_5 = np.zeros((1 * length), dtype=np.long)
        multi_labels_6 = np.zeros((1 * length), dtype=np.long)
        multi_labels_7 = np.zeros((1 * length), dtype=np.long)
        multi_labels_8 = np.zeros((1 * length), dtype=np.long)
        multi_labels_9 = np.zeros((1 * length), dtype=np.long)
        for im_id in range(length):
            for rim in range(1):
                multi_labels_1[1*im_id+rim] = parsed[im_id]
                multi_labels_2[1*im_id+rim] = parsed[extension[1*im_id+rim]]
                multi_labels_3[1 * im_id + rim] = parsed[extension2[1 * im_id + rim]]
                multi_labels_4[1 * im_id + rim] = parsed[extension3[1 * im_id + rim]]
                multi_labels_5[1 * im_id + rim] = parsed[extension4[1 * im_id + rim]]
                multi_labels_6[1 * im_id + rim] = parsed[extension5[1 * im_id + rim]]
                multi_labels_7[1 * im_id + rim] = parsed[extension6[1 * im_id + rim]]
                multi_labels_8[1 * im_id + rim] = parsed[extension7[1 * im_id + rim]]
                multi_labels_9[1 * im_id + rim] = parsed[extension8[1 * im_id + rim]]
        return torch.from_numpy(parsed).view(length).long(), torch.from_numpy(multi_labels_1).view(length*1).long(), \
               torch.from_numpy(multi_labels_2).view(length*1).long(), torch.from_numpy(multi_labels_3).view(length*1).long(), \
               torch.from_numpy(multi_labels_4).view(length*1).long(), torch.from_numpy(multi_labels_5).view(length*1).long(), \
               torch.from_numpy(multi_labels_6).view(length * 1).long(), torch.from_numpy(multi_labels_7).view(length*1).long(), \
               torch.from_numpy(multi_labels_8).view(length * 1).long(), torch.from_numpy(multi_labels_9).view(length*1).long()



def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        pv = parsed.reshape(length, num_rows, num_cols)
        multi_length = length * 1
        multi_data = np.zeros((1*length, num_rows, num_cols))
        extension = np.zeros(1*length, dtype=np.int32)
        extension2 = np.zeros(1 * length, dtype=np.int32)
        extension3 = np.zeros(1 * length, dtype=np.int32)
        extension4 = np.zeros(1 * length, dtype=np.int32)
        extension5 = np.zeros(1 * length, dtype=np.int32)
        extension6 = np.zeros(1 * length, dtype=np.int32)
        extension7 = np.zeros(1 * length, dtype=np.int32)
        extension8 = np.zeros(1 * length, dtype=np.int32)
        for left in range(length):
            chosen_ones = np.random.permutation(length)[:1]
            chosen_extras = np.random.permutation(length)[:1]
            chosen_extras_2 = np.random.permutation(length)[:1]
            chosen_extras_3 = np.random.permutation(length)[:1]
            chosen_extras_4 = np.random.permutation(length)[:1]
            chosen_extras_5 = np.random.permutation(length)[:1]
            chosen_extras_6 = np.random.permutation(length)[:1]
            chosen_extras_7 = np.random.permutation(length)[:1]
            extension[left*1:(left+1)*1] = chosen_ones
            extension2[left * 1:(left + 1) * 1] = chosen_extras
            extension3[left * 1:(left + 1) * 1] = chosen_extras_2
            extension4[left * 1:(left + 1) * 1] = chosen_extras_3
            extension5[left * 1:(left + 1) * 1] = chosen_extras_4
            extension6[left * 1:(left + 1) * 1] = chosen_extras_5
            extension7[left * 1:(left + 1) * 1] = chosen_extras_6
            extension8[left * 1:(left + 1) * 1] = chosen_extras_7
            for j, (right, extra, extra2, extra3, extra4, extra5, extra6, extra7) in enumerate(zip(chosen_ones, chosen_extras, chosen_extras_2, chosen_extras_3, chosen_extras_4, chosen_extras_5, chosen_extras_6, chosen_extras_7)):
                l1im = pv[left,:,:]
                l2im = pv[right,:,:]
                l3im = pv[extra, :, :]
                l4im = pv[extra2, :, :]
                l5im = pv[extra3, :, :]
                l6im = pv[extra4, :, :]
                l7im = pv[extra5, :, :]
                l8im = pv[extra6, :, :]
                l9im = pv[extra7, :, :]
                new_im = np.zeros((52, 52))
                new_im[0:28,0:28] = l1im
                new_im[12:40,0:28] = l2im
                new_im[24:52, 0:28] = l3im
                new_im[0:28, 12:40] = l4im
                new_im[12:40, 12:40] = l5im
                new_im[24:52, 12:40] = l6im
                new_im[0:28, 24:52] = l7im
                new_im[12:40, 24:52] = l8im
                new_im[24:52, 24:52] = l9im

                new_im[12:24, 0:12] = np.maximum(l1im[12:24, 0:12], l2im[0:12, 0:12])
                new_im[24:28, 0:12] = np.maximum(np.maximum(l1im[24:28, 0:12], l2im[12:16, 0:12]), l3im[0:4, 0:12])
                new_im[28:40, 0:12] = np.maximum(l2im[16:28, 0:12], l3im[4:16, 0:12])

                new_im[0:12, 12:24] = np.maximum(l1im[0:12, 12:24], l4im[0:12, 0:12])
                new_im[12:24, 12:24] = np.maximum(np.maximum(np.maximum(l1im[12:24, 12:24], l2im[0:12, 12:24]), l4im[12:24, 0:12]), l5im[0:12, 0:12])
                new_im[24:28, 12:24] = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(l1im[24:28, 12:24], l2im[12:16, 12:24]), l3im[0:4, 12:24]), l4im[24:28, 0:12]), l5im[12:16, 0:12]), l6im[0:4, 0:12])
                new_im[28:40, 12:24] = np.maximum(np.maximum(np.maximum(l2im[16:28, 12:24], l3im[4:16, 12:24]), l5im[16:28, 0:12]), l6im[4:16, 0:12])
                new_im[40:52, 12:24] = np.maximum(l3im[16:28, 12:24], l6im[16:28, 0:12])

                new_im[0:12, 24:28] = np.maximum(np.maximum(l1im[0:12, 24:28], l4im[0:12, 12:16]), l7im[0:12, 0:4])
                new_im[12:24, 24:28] = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(l1im[12:24, 24:28], l2im[0:12, 24:28]), l4im[12:24, 12:16]), l5im[0:12, 12:16]), l7im[12:24, 0:4]), l8im[0:12, 0:4])
                new_im[24:28, 24:28] = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(l1im[24:28, 24:28], l2im[12:16, 24:28]), l3im[0:4, 24:28]), l4im[24:28, 12:16]), l5im[12:16, 12:16]), l6im[0:4, 12:16]), l7im[24:28, 0:4]), l8im[12:16, 0:4]), l9im[0:4, 0:4])
                new_im[28:40, 24:28] = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(l2im[16:28, 24:28], l3im[4:16, 24:28]), l5im[16:28, 12:16]), l6im[4:16, 12:16]), l8im[16:28, 0:4]), l9im[4:16, 0:4])
                new_im[40:52, 24:28] = np.maximum(np.maximum(l3im[16:28, 24:28], l6im[16:28, 12:16]), l9im[16:28, 0:4])

                new_im[0:12, 28:40] = np.maximum(l4im[0:12, 16:28], l7im[0:12, 4:16])
                new_im[12:24, 28:40] = np.maximum(np.maximum(np.maximum(l4im[12:24, 16:28], l5im[0:12, 16:28]), l7im[12:24, 4:16]), l8im[0:12, 4:16])
                new_im[24:28, 28:40] = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(l4im[24:28, 16:28], l5im[12:16, 16:28]), l6im[0:4, 16:28]), l7im[24:28, 4:16]), l8im[12:16, 4:16]), l9im[0:4, 4:16])
                new_im[28:40, 28:40] = np.maximum(np.maximum(np.maximum(l5im[16:28, 16:28], l6im[4:16, 16:28]), l8im[16:28, 4:16]), l9im[4:16, 4:16])
                new_im[40:52, 28:40] = np.maximum(l6im[16:28, 16:28], l9im[16:28, 4:16])

                new_im[12:24, 40:52] = np.maximum(l7im[12:24, 16:28], l8im[0:12, 16:28])
                new_im[24:28, 40:52] = np.maximum(np.maximum(l7im[24:28, 16:28], l8im[12:16, 16:28]), l9im[0:4, 16:28])
                new_im[28:40, 40:52] = np.maximum(l8im[16:28, 16:28], l9im[4:16, 16:28])

                #multi_data_im =  m.imresize(new_im, (28, 28), interp='nearest')
                multi_data_im = np.array(Image.fromarray(new_im).resize((28, 28)))
                multi_data[left*1 + j,:,:] = multi_data_im
        return torch.from_numpy(parsed).view(length, num_rows, num_cols), torch.from_numpy(multi_data).view(length,num_rows, num_cols), extension, extension2, extension3, extension4, extension5, extension6, extension7, extension8

if __name__ == '__main__':
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import matplotlib.pyplot as plt

    def global_transformer():
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])

    dst = MNIST(root='/home/ozansener/Data/MultiMNIST/', train=True, download=True, transform=global_transformer(), multi=True)
    loader = torch.utils.data.DataLoader(dst, batch_size=10, shuffle=True, num_workers=4)
    for dat in loader:
        ims = dat[0].view(10,28,28).numpy()

        labs_l = dat[1]
        labs_r = dat[2]
        f, axarr = plt.subplots(2,5)
        for j in range(5):
            for i in range(2):
                axarr[i][j].imshow(ims[j*2+i,:,:], cmap='gray')
                axarr[i][j].set_title('{}_{}'.format(labs_l[j*2+i],labs_r[j*2+i]))
        plt.show()
        a = input()
        if a == 'ex':
            break
        else:
            plt.close()


