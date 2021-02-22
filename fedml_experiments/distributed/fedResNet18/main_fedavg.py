import argparse
import logging
import os
import random
import socket

import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

# add the FedML root directory to the python path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.model.ResNet18.model import Net
from fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed
from fedml_api.model.ResNet18.ResNet18Trainer import ResNet18Trainer
from fedml_api.model.ResNet18.ResNet18ResultAggregator import ResNet18ResultAggregator

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',help='number of workers in a distributed cluster')
    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN', help='number of workers')
    parser.add_argument('--comm_round', type=int, default=10, help='how many round of communications we shoud use')
    parser.add_argument('--is_mobile', type=int, default=0, help='whether the program is running on the FedML-Mobile server side')
    parser.add_argument('--frequency_of_the_test', type=int, default=1, help='the frequency of the algorithms')
    parser.add_argument('--gpu_server_num', type=int, default=1, help='gpu_server_num')
    parser.add_argument('--gpu_num_per_server', type=int, default=1, help='gpu_num_per_server')

    # Model configurations
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--epochs', default=10, type=int) #100
    parser.add_argument('--batches', default=20, type=int) #500
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument("--data_augment", action='store_true')
    parser.add_argument('--data_path', default='/FineFL/fedml_api/model/ResNet18/cifar10/', type=str)
    args = parser.parse_args()
    return args

def create_model(args):
    logging.info("ResNet18 + cifar10")
    if args.model == 'toy':
        model = Net()
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    else:
        raise NotImplementedError
    return model

def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    return device


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(
            # project="federated_nas",
            project="fedml",
            name="FedAVG(d)" + "r" + str(args.comm_round) + "-e" + str(args.epochs),
            config=args
        )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # GPU arrangement: Please customize this function according your own topology.
    # The GPU server list is configured at "mpi_host_file".
    # If we have 4 machines and each has two GPUs, and your FL network has 8 workers and a central worker.
    # The 4 machines will be assigned as follows:
    # machine 1: worker0, worker4, worker8;
    # machine 2: worker1, worker5;
    # machine 3: worker2, worker6;
    # machine 4: worker3, worker7;
    # Therefore, we can see that workers are assigned according to the order of machine list.
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)

    # TODO
    # define train_path and test_path
    train_path = os.path.join(args.data_path, "client_" + str(process_id) + "_train")
    test_path = os.path.join(args.data_path, "client_" + str(process_id) + "_test")

    # create model.
    model = create_model(args)
    
    # start "federated averaging (FedAvg)"
    model_trainer = ResNet18Trainer(model)
    result_aggregator = ResNet18ResultAggregator()
    FedML_FedAvg_distributed(process_id, worker_number, device, comm, model, args, train_path, test_path, result_aggregator, model_trainer)