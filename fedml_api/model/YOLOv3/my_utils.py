from torch.autograd import Variable
from terminaltables import AsciiTable
import torch
from fedml_api.model.YOLOv3.utils.logger import *
import time
from fedml_api.model.YOLOv3.utils.utils import *
from fedml_api.model.YOLOv3.test import evaluate


class My_dataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            try:
                print('Begin one epoch, iter', len(self.dataloader))
                for item in self.dataloader:
                    yield item
            except StopIteration:
                print('Finish one epoch, iter')

    def __next__(self):
        while True:
            try:
                print('Begin one epoch, iter', len(self.dataloader))
                for item in self.dataloader:
                    return item
            except StopIteration:
                print('Finish one epoch, next')


def train_model(device, metrics, batch_id, model, optimizer, my_iter, logger):
    model.train()
    (_, imgs, targets) = next(my_iter)
    imgs = Variable(imgs.to(device))
    targets = Variable(targets.to(device), requires_grad=False)
    loss, outputs = model(imgs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # ----------------
    #   Log progress
    # ----------------

    log_str = "\n---- [Batch %d] ----\n" % batch_id

    metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

    # Log metrics at each YOLO layer
    for i, metric in enumerate(metrics):
        formats = {m: "%.6f" for m in metrics}
        formats["grid_size"] = "%2d"
        formats["cls_acc"] = "%.2f%%"
        row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
        metric_table += [[metric, *row_metrics]]

        # Tensorboard logging
        tensorboard_log = []
        for j, yolo in enumerate(model.yolo_layers):
            for name, metric in yolo.metrics.items():
                if name != "grid_size":
                    tensorboard_log += [(f"train/{name}_{j + 1}", metric)]
        tensorboard_log += [("train/loss", to_cpu(loss).item())]
        logger.list_of_scalars_summary(tensorboard_log, 0)

    log_str += AsciiTable(metric_table).table
    log_str += f"\nTotal loss {to_cpu(loss).item()}"
    model.seen += imgs.size(0)
    return log_str


def eval_model(class_names, model, valid_path, opt, logger):
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=0.0,
        conf_thres=0.0,
        nms_thres=0.0,
        img_size=opt.img_size,
        batch_size=8,
    )
    evaluation_metrics = [
        ("validation/precision", precision.mean()),
        ("validation/recall", recall.mean()),
        ("validation/mAP", AP.mean()),
        ("validation/f1", f1.mean()),
    ]
    logger.list_of_scalars_summary(evaluation_metrics, 0)

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")

if __name__ == "__main__":
    a = [1, 2, 3]
    b = My_dataloader(a)
    iter = b.__iter__()
    for i in range(10):
        a = next(iter)
        print(i, a)