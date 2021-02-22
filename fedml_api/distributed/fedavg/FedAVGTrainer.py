from .utils import transform_tensor_to_list
import os
import logging
import json
class FedAVGTrainer(object):

    def __init__(self, client_index, device, args, train_path, test_path, model_trainer):
        self.trainer = model_trainer
        self.client_index = client_index
        self.device = device
        self.args = args
        self.train_path = train_path
        self.test_path = test_path
        self.local_sample_number = self.trainer.get_local_sample_number(self.train_path)
        
    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def train(self):
        self.trainer.train(self.device, self.args, self.train_path)
        weights = self.trainer.get_model_params()
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number

    def test(self):
        logging.info("testing for data {}".format(self.test_path))
        result = self.trainer.test(self.device, self.args, self.test_path)
        return json.dumps(result), self.local_sample_number