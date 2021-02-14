import csv


class Predictor(object):
    def __init__(self, config, logger, model, preprocessor):
        self.config = config
        self.logger = logger
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, x_test):
        pass

    def save_result(self, test_ids, y_prob_pred):
        pass