import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig


class LinearNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.__create_model()

    def forward(self, input_ids, attention_mask):
        # roBERTa
        #    input_ids/attention_mask dimensions - [-1, sequence_length=128, hidden_size=768]
        #    os dimensions -  [-1, sequence_length=128, hidden_size=768]
        #    ms dimensions - [-1, hidden_size=768]
        #    hs tuple dimensions - 13*[-1, sequence_length=128, hidden_size=768]
        os, ms, hs = self.roberta(input_ids, attention_mask)
        x = torch.stack([hs[-1], hs[-2], os])
        x = torch.mean(x, 0)

        # Dropout
        x = self.dropout(x)

        # Linear
        x = self.fc(x)

        # Outputs Logits
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

    def __create_model(self):
        # roBERTa
        roberta_config = RobertaConfig.from_pretrained(self.config['roberta_config_file_path'], output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained(self.config['model_file_path'], config=roberta_config)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Linear
        self.fc = nn.Linear(roberta_config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)