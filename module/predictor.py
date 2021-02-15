import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from module.model import LinearNN
from module.trainer import Evaluation_Function


class Predictor(object):
    def __init__(self, config, logger, preprocessor):
        self.config = config
        self.logger = logger
        self.preprocessor = preprocessor
        self.models = self.__load_models()
        self.predictions = []

    def predict(self):
        for data in tqdm(self.preprocessor.test_loader):
            ids, masks, tweet, offsets = Predictor.__unpack_data(data)

            start_logits, end_logits = [], []
            for model in self.models:
                with torch.no_grad():
                    output = model(ids, masks)
                    start_logits.append(torch.softmax(output[0], dim=1).cpu().detach().numpy())
                    end_logits.append(torch.softmax(output[1], dim=1).cpu().detach().numpy())

            start_logits, end_logits = np.mean(start_logits, axis=0), np.mean(end_logits, axis=0)
            for i in range(len(ids)):
                start_pred, end_pred = np.argmax(start_logits[i]), np.argmax(end_logits[i])
                if start_pred > end_pred:
                    pred = tweet[i]
                else:
                    pred = Evaluation_Function.get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
                self.predictions.append(pred)

    def save_result(self):
        selected_texts = []
        for index, row in self.preprocessor.test_df.iterrows():
            text = row.text
            output_str = ""
            if row.sentiment == 'neutral' or len(text.split()) <= 2:
                selected_texts.append(text)
            else:
                selected_texts.append(self.predictions[index])

        sub_df = pd.read_csv(self.config['sample_submission_file_path'])
        sub_df['selected_text'] = selected_texts
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split()) == 1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split()) == 1 else x)
        sub_df['selected_text'] = sub_df['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split()) == 1 else x)
        sub_df.to_csv(self.config['submission_file_path'], index=False)

    def __load_models(self):
        models = []
        for root, dirs, files in os.walk(self.config['model_output_file'], topdown=False):
            for name in files:
                self.logger.info(f'Loading Model: {name}')
                model = LinearNN(self.config)
                if torch.cuda.is_available():
                    model.cuda()
                    parameters = torch.load(os.path.join(root, name))
                else:
                    model.cpu()
                    parameters = torch.load(os.path.join(root, name), map_location=torch.device('cpu'))
                model.load_state_dict(parameters)
                model.eval()
                models.append(model)
        return models

    @staticmethod
    def __unpack_data(batch_data):
        if torch.cuda.is_available():
            ids = batch_data['ids'].cuda()
            masks = batch_data['masks'].cuda()
            tweet = batch_data['tweet']
            offsets = batch_data['offsets'].numpy()
        else:
            ids = batch_data['ids'].cpu()
            masks = batch_data['masks'].cpu()
            tweet = batch_data['tweet']
            offsets = batch_data['offsets'].numpy()
        return ids, masks, tweet, offsets