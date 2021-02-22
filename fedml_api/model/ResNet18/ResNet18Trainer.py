from __future__ import division

import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import tqdm
from fedml_core.trainer.model_trainer import ModelTrainer
from fedml_api.model.ResNet18.model import Net
from fedml_api.model.ResNet18.utils import *

class ResNet18Trainer(ModelTrainer):
    def get_model_params(self):
        return get_model_parameters(self.model)
    def set_model_params(self, model_parameters):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_model_parameters(self.model, model_parameters, device)
    def get_local_sample_number(self, train_path):
        return 1
    def train(self, device, args, train_path):
        if args.data_augment:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            trainset = ImageDataset(train_path, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            trainset = ImageDataset(train_path, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # Get data configuration
        model = self.model
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        fl_train_loader = My_dataloader(trainloader)
        fl_iter = fl_train_loader.__iter__()

        for epoch_id in range(args.epochs):
            running_loss = 0.0
            for batch_id in tqdm.trange(args.batches):
                loss = train_model(fl_iter, optimizer, model, criterion, device)
                running_loss += loss.item()
            # print statistics
            print('[Epoch: %5d] loss: %.3f' % (epoch_id, running_loss / args.batches))
    
    def test(self, device, args, test_path):
        model = self.model
        model = model.to(device)
        if args.data_augment:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            testset = ImageDataset(test_path, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            testset = ImageDataset(test_path, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        accuracy = eval_model(testloader, self.model, device)
        return {"Accuracy": accuracy}