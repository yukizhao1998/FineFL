import torch
import random
import numpy as np
import os
import torch.utils.data as data
import pickle
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_model_parameters(model):
    with torch.no_grad():
        param_dict = {}
        for name, param in model.named_parameters():
            param_dict[name] = param.data.cpu().numpy()
        return param_dict


def set_model_parameters(model, param_dict, device):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = torch.tensor(param_dict[name]).to(device)


class My_dataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            try:
                for item in self.dataloader:
                    yield item
            except StopIteration:
                pass


def train_model(fl_iter, optimizer, model, criterion, device):
    model.train()
    images, labels = next(fl_iter)
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def eval_model(data_loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return 1.0 * correct / total

def eval_model_classes(data_loader, model, classes, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


class ImageDataset(data.Dataset):
    def __init__(
            self,
            root,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        print(self.root)
        self.data = pickle.load(open(self.root + '_data', 'rb'))
        self.targets = pickle.load(open(self.root + '_target', 'rb'))
        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)