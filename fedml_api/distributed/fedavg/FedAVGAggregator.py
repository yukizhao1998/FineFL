import copy
import logging
import time

import numpy as np
import wandb

from .utils import transform_list_to_tensor


class FedAVGAggregator(object):

    def __init__(self, client_num, device, args, result_aggregator, model_trainer):
        self.trainer = model_trainer
        self.result_aggregator = result_aggregator
        self.client_num = client_num
        self.device = device
        self.args = args
        self.client_indexes = []
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.flag_client_result_received_dict = dict()
        self.test_result = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def set_client_indexes(self, client_indexes):
        self.client_indexes = client_indexes

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in self.client_indexes:
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0
        
        for idx in self.client_indexes:
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def add_test_result(self, index, result, sample_number, round_idx):
        logging.info("add test result {}, {}".format(round_idx, index))
        if round_idx not in self.test_result.keys():
            self.test_result[round_idx] = {}
            self.flag_client_result_received_dict[round_idx]={}
            for i in range(self.client_num):
                self.flag_client_result_received_dict[round_idx][i] = False
        self.test_result[round_idx][index] = {"result":result, "sample_number":sample_number}
        self.flag_client_result_received_dict[round_idx][index] = True

    def check_whether_all_result_receive(self, round_idx):
        if round_idx not in self.flag_client_result_received_dict.keys():
            return False
        '''
        for idx in self.client_indexes:
        '''
        for idx in range(self.client_num):
            if not self.flag_client_result_received_dict[round_idx][idx]:
                return False
        return True

    # might vary with tasks
    def aggregate_test_result(self, round_idx):
        result = self.result_aggregator.aggregate_test_result(round_idx, self.test_result[round_idx])
        return result