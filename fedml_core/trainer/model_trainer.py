from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, model):
        self.model = model
        self.id = 0

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def get_local_sample_number(self, data_path):
        pass

    @abstractmethod
    def train(self, device, args, data_path):
        pass

    @abstractmethod
    def test(self, device, args, data_path):
        pass
