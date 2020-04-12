import torch
from torchvision import transforms

from .multi_mnist_loader_extratask9 import MNIST_ETASK9


mnist_etask9_path = "/home/hilalevi/data/Datasets/MultiMnistEtask9"

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(params):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'mnist_etask9' in params['dataset']:
        train_dst = MNIST_ETASK9(root=mnist_etask9_path, train=True, download=True, transform=global_transformer(), multi=True)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)

        val_dst = MNIST_ETASK9(root=mnist_etask9_path, train=False, download=True, transform=global_transformer(), multi=True)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=4)

        return train_loader, train_dst, val_loader, val_dst

