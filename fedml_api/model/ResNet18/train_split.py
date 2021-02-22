import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import tqdm
from model import Net
import pickle
from utils import *


def main(conf):
    if conf.data_augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        trainset = ImageDataset('./split_data/client_0_train', transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, )
        testset = ImageDataset('./split_data/client_0_test', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=conf.batch_size, shuffle=False)
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = ImageDataset('./split_data/client_0_train', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_size, shuffle=True)
        testset = ImageDataset('./split_data/client_0_test', transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=conf.batch_size, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if conf.model == 'toy':
        model = Net()
    elif conf.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    else:
        raise NotImplementedError
    model = model.to(conf.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fl_train_loader = My_dataloader(trainloader)
    fl_iter = fl_train_loader.__iter__()

    for epoch_id in range(conf.epochs):
        running_loss = 0.0
        for batch_id in tqdm.trange(conf.batches):
            loss = train_model(fl_iter, optimizer, model, criterion, conf.device)
            running_loss += loss.item()

        # print statistics
        print('[Epoch: %5d] loss: %.3f' % (epoch_id, running_loss / conf.batches))

        # perform synchronize
        model_parameters = get_model_parameters(model)
        print('Synchronize')
        set_model_parameters(model, model_parameters, conf.device)

        if epoch_id % conf.evaluation_interval == 0:
            eval_model(testloader, model, conf.device)

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment setup')
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batches', default=500, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument("--evaluation_interval", default=1, type=int)
    parser.add_argument("--data_augment", action='store_true')
    conf = parser.parse_args()

    assert conf.model in ['resnet18', 'toy']

    if torch.cuda.is_available():
        conf.device = 'cuda'
    else:
        conf.device = 'cpu'
    print(conf.__dict__)
    main(conf)


