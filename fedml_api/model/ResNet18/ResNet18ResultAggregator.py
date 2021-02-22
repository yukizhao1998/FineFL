import logging

class ResNet18ResultAggregator():
    def __init__(self):
        self.round_idx = 0
    def set_round_num(round_idx):
        self.round_idx = round_idx
    def aggregate_test_result(self, round_idx, result_dict):
        logging.info("round {} aggregating result...".format(round_idx))
        tags = result_dict[0]["result"].keys()
        total_data_cnt = 0
        total_result = result_dict[0]["result"]
        for idx in result_dict.keys():
            for tag in tags:
                total_data_cnt += result_dict[idx]["sample_number"]
        total_result = result_dict[0]["result"]
        for idx in result_dict.keys():
            for tag in tags:
                total_result[tag] += result_dict[idx]["result"][tag] * result_dict[idx]["sample_number"]
        for tag in tags:
            total_result[tag] /= total_data_cnt
        logging.info(total_result)
        return total_result