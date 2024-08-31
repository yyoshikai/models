import pickle
import numpy as np
import torch
from models.data import DataLoader
from models.accumulator2 import Accumulator
from models.metric import get_metric
from models import Model

class Validator:
    def __init__(self, dl, process,
            accumulator={}, metrics=[]):
        self.dl = DataLoader(**dl)
        self.metrics = [get_metric(m) for m in metrics]
        self.accumulator = Accumulator(**accumulator)
        self.process = process
        pass

    def evaluate(self, model: Model, batch: dict={}, save_agg_dir=None):
        training = model.training
        model.eval()

        # initialization
        self.accumulator.init()
        for m in self.metrics:
            m.init()

        # validation
        with torch.inference_mode():
            for batch0 in self.dl:
                model(batch0, self.process)
                self.accumulator(batch0)
                for m in self.metrics:
                    m(batch0)
        
        if training: model.train()

        # Result
        ## metrics
        scores = {}
        for m in self.metrics:
            m.calc(scores)
        batch.update(scores)
        ## accumulator
        if save_agg_dir is not None:
            accum = self.accumulator.save_agg(save_agg_dir)
        else:
            accum = self.accumulator.agg()
        return scores, accum





