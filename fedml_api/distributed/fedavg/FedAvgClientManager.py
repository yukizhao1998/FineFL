import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor


class FedAVGClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_TEST_MODEL_ON_CLIENT,
                                              self.handle_message_test_model_on_client)

    def handle_message_test_model_on_client(self, msg_params):
        #logging.info("client {} receive test request".format(self.rank - 1))
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)
        self.trainer.update_model(model_params)
        logging.info("client {} receive test request {}".format(self.rank - 1, msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)))
        self.__test(msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX))
        logging.info("client {} finish testing for {} round".format(self.rank - 1, msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)))
        '''
        if msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX) == self.num_rounds - 1:
            self.finish()
        '''

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)
        self.trainer.update_model(global_model_params)
        self.__train()

    '''
    def start_training(self):
        self.__train()
    '''

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)
        self.trainer.update_model(model_params)
        self.__train()
        

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_test_result_to_server(self, receive_id, test_result, local_sample_num, round_idx):
        logging.info("client {} sending test result".format(self.get_sender_id() - 1))
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_TEST_RESULT_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_TEST_RESULT, test_result)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### client = {}".format(self.get_sender_id() - 1))
        weights, local_sample_num = self.trainer.train()
        self.send_model_to_server(0, weights, local_sample_num)

    def __test(self, round_idx):
        test_result, local_sample_num = self.trainer.test()
        self.send_test_result_to_server(0, test_result, local_sample_num, round_idx)