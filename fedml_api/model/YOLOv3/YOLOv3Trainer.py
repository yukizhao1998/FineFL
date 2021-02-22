from __future__ import division

from fedml_api.model.YOLOv3.models import *
from fedml_api.model.YOLOv3.utils.logger import *
from fedml_api.model.YOLOv3.utils.utils import *
from fedml_api.model.YOLOv3.utils.datasets import *
from fedml_api.model.YOLOv3.utils.augmentations import *
from fedml_api.model.YOLOv3.utils.transforms import *
from fedml_api.model.YOLOv3.utils.parse_config import *

from terminaltables import AsciiTable
import logging
import os
import sys
import time
import datetime
import argparse
import tqdm
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from fedml_api.model.YOLOv3.my_utils import train_model, eval_model, My_dataloader
from fedml_core.trainer.model_trainer import ModelTrainer

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    cnt = 0
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        cnt += 1
        if cnt == 5:
            break
        if targets is None:
            continue
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return precision, recall, AP, f1, ap_class

class YOLOv3Trainer(ModelTrainer):
    def get_model_params(self):
        return self.model.get_model_parameters()
    def set_model_params(self, model_parameters):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.model.set_model_parameters(model_parameters, device)
    def get_local_sample_number(train_path):
        return len(os.listdir(train_path))
    def train(self, device, args, train_path):
        logger = Logger(args.logdir)
        # Get data configuration
        model=self.model
        model.to(device)
        # Get dataloader
        dataset = ListDataset(train_path, multiscale=args.multiscale_training, transform=AUGMENTATION_TRANSFORMS)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        optimizer = torch.optim.Adam(model.parameters())

        metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]

        my_dataloader = My_dataloader(dataloader)
        my_iter = my_dataloader.__iter__()

        for epoch in range(args.epochs):
            print('\n---- [Epoch %d/%d] ----\n' % (epoch + 1, args.epochs))
            for batch_i in range(103):
                log_str = train_model(device, metrics, batch_i, model, optimizer, my_iter, logger)
                if args.verbose and batch_i % 100 == 0: print(log_str)
            '''
            if epoch % opt.evaluation_interval == 0:
                eval_model(class_names, model, valid_path, opt, logger)

            if epoch % opt.checkpoint_interval == 0:
                torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
            '''
    
    def test(self, device, args, test_path):
        logger = Logger(args.logdir)
        model = self.model
        model.to(device)
        print("Compute mAP...")
        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=test_path,
            iou_thres=args.iou_thres,
            conf_thres=args.conf_thres,
            nms_thres=args.nms_thres,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )
        result_dict = {"precision":list(precision), "recall":list(recall), "AP":list(AP), "f1": list(f1)}
        for tag in result_dict:
            result_dict[tag] = [1.0 * i for i in result_dict[tag]]
        logging.info(result_dict)
        return json.dumps(result_dict)