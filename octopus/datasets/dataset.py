import torch
from torchvision import datasets, transforms

def gen_data_loader(problem, adv_train, batch_size, test_batch_size, use_cuda):

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = eval(f'datasets.{problem}')('data', train=True, download=True,
                       transform=transform)
    dataset2 = eval(f'datasets.{problem}')('data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


# adversarial training PGD
def pgd():
    pass

def VAE():
    pass