import torch

from Model import AttentionModel


class Baseline(object):
    def __init__(self, model: AttentionModel, graph_size, nb_val_samples):
        self.model = model
        self.graph_size = graph_size
        self.nb_val_samples = nb_val_samples
        self.tour_lengths = []

    def eval(self):
        with torch.no_grad():
            return self.model.eval()

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def evaluate(self, inputs, use_solver=False):
        if use_solver:
            raise NotImplementedError("Solver doesnt have a stable implementation yet.")
        _, _, rollout_sol = self.model(inputs)
        self.tour_lengths = self.model.splitNumba(inputs, rollout_sol)
        return self.tour_lengths

    def set_decode_mode(self, mode):
        self.model.set_decode_mode(mode)

    def reset(self):
        self.tour_lengths = []
