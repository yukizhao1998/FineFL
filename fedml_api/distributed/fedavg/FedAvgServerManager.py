import logging
import os
import sys
import json
import time
from .message_define import MyMessage
from .utils import transform_tensor_to_list

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FineFL")))
try:
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager

class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        self.aggregator.set_client_indexes(client_indexes)
        for client_index in client_indexes:
            receive_id = client_index + 1
            self.send_message_init_config(receive_id, global_model_params)


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_TEST_RESULT_TO_SERVER,
                                              self.handle_message_receive_result_from_client)
    
    def start_training_round(self):
        #start next round
        global_model_params = self.aggregator.get_global_model_params()
        self.round_idx += 1
        if self.round_idx == self.round_num:
            self.finish()
            return  
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                        self.args.client_num_per_round)
        self.aggregator.set_client_indexes(client_indexes)
        if self.args.is_mobile == 1:
            print("transform_tensor_to_list")
            global_model_params = transform_tensor_to_list(global_model_params)

        for client_index in client_indexes:
            receiver_id = client_index + 1
            self.send_message_sync_model_to_client(receiver_id, global_model_params)

    def handle_message_receive_result_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        result = json.loads(msg_params.get(MyMessage.MSG_ARG_KEY_TEST_RESULT))
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        round_idx = msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)
        
        self.aggregator.add_test_result(sender_id - 1, result, local_sample_number, round_idx)
        result_all_received = self.aggregator.check_whether_all_result_receive(round_idx)
        if result_all_received:
            logging.info("aggregating result for the " + str(round_idx) + " th round...")
            test_result = self.aggregator.aggregate_test_result(round_idx)
            print(test_result)
            self.start_training_round()
            

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()
            # test on all clients(last trained clients)
            if self.round_idx % self.args.frequency_of_the_test == 0 or self.round_idx == self.round_num - 1:
                
                for receiver_id in range(1, self.size):
                    self.send_message_test_model_on_clients(receiver_id, global_model_params, self.round_idx)
                '''
                for client_index in self.aggregator.client_indexes:
                    receiver_id = client_index + 1
                    self.send_message_test_model_on_clients(receiver_id, global_model_params, self.round_idx)
                '''
            else:
                self.start_training_round()

    def send_message_init_config(self, receive_id, global_model_params):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        self.send_message(message)

    def send_message_test_model_on_clients(self, receiver_id, global_model_params, round_idx):
        logging.info("sending test request to receiver {}".format(receiver_id - 1))
        message = Message(MyMessage.MSG_TYPE_S2C_TEST_MODEL_ON_CLIENT, self.get_sender_id(), receiver_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        self.send_message(message)