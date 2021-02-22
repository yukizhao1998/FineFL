class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2
    MSG_TYPE_S2C_TEST_MODEL_ON_CLIENT = 3
    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 4
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 5
    MSG_TYPE_C2S_SEND_TEST_RESULT_TO_SERVER = 6

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"
    MSG_ARG_KEY_TEST_RESULT = "test_result"
    MSG_ARG_KEY_ROUND_IDX = "round_idx"