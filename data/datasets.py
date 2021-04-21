import torch
from torchvision import transforms

from .multi_mnist_loader_extratask9 import MNIST_ETASK9
from .clevr_loader import ClevrDatasetImagesStateDescription, my_collate


mnist_etask9_path = "/home/hilalevi/data/Datasets/MultiMnistEtask9"
clevr_path = '/home/hilalevi/data/Datasets/clevr/CLEVR_v1.0/CLEVR_v1.0/'

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


    if 'clevr' in params['dataset']:
        transform_clevr = transforms.Compose([
            transforms.Resize((224, 224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = ClevrDatasetImagesStateDescription(clevr_path, train=True, transform=transform_clevr)
        val_dataset = ClevrDatasetImagesStateDescription(clevr_path, train=False, transform=transform_clevr)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=8)  # , collate_fn=my_collate)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], num_workers=8)  # , collate_fn=my_collate)
        return train_loader, train_dataset, val_loader, val_dataset
    
