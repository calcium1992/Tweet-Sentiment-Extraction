import pandas as pd
import torch
import tokenizers
from sklearn.model_selection import train_test_split


class TweetDataset(torch.utils.data.Dataset):
    # Description:
    #    Inherit from torch.utils.data.Dataset,
    #    which is an abstract class representing a dataset. All subclasses should
    #    overwrite __getitem__ to map index into each sample and optionally overwrite
    #    __len__ to get number of samples.
    #    (Ref. "https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset")
    def __init__(self, data_df, config):
        self.data_df, self.config = data_df, config
        self.maxlen = self.config['maxlen']
        self.labeled = 'selected_text' in data_df
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=self.config['vocab_file_path'],
            merges_file=self.config['merge_file_path'],
            lowercase=True,
            add_prefix_space=True)

    def __getitem__(self, index):
        # Extract One Row/Sample
        data = {}
        row = self.data_df.iloc[index]

        # Get One Sample using []
        ids, masks, tweet, offsets = self.__get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets
        if self.labeled:  # This is only for training data.
            start_idx, end_idx = self.__get_target_idx(row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx

        return data

    def __len__(self):
        return len(self.data_df)

    def __get_input_data(self, row):
        # Calculate ids
        #    ids = <s> + sentiment_id + </s> + </s> + encoding_ids + </s>
        #    sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]

        # Calculate offsets
        #    Offsets are the index of start chars and end chars in each sentence. It
        #    only shows for sentences, not <s> and </s>. This is not model inputs.
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

        # Pad
        pad_len = self.maxlen - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len  # Pad with 1
            offsets += [(0, 0)] * pad_len  # Pad with (0, 0)

        # Translate ids, masks, offsets into Tensors
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)

        return ids, masks, tweet, offsets

    def __get_target_idx(self, row, tweet, offsets):
        # ???How to implement to get start index and end index???
        # Situation like "sooo sad" into [so, oo, sad] is tricky.

        selected_text = " " + " ".join(row.selected_text.lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]

        return start_idx, end_idx


class Preprocessor(object):
    def __init__(self, config, logger):
        self.config, self.logger = config, logger
        self.__read_files()
        self.train_loader, self.val_loader, self.test_loader = None, None, None

    def generate_data_loaders(self):
        train_df, val_df, _, _ = train_test_split(self.train_val_df, self.train_val_df['sentiment'], test_size=0.2)
        self.train_loader = torch.utils.data.DataLoader(TweetDataset(train_df, self.config),
                                                        batch_size=self.config['batch_size'], shuffle=True,
                                                        num_workers=0, drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(TweetDataset(val_df, self.config),
                                                      batch_size=self.config['batch_size'], shuffle=False,
                                                      num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(TweetDataset(self.test_df, self.config),
                                                       batch_size=self.config['batch_size'], shuffle=False,
                                                       num_workers=0)

    def __read_files(self):
        self.train_val_df = pd.read_csv(self.config['training_file_path'])
        self.train_val_df['text'] = self.train_val_df['text'].astype(str)
        self.train_val_df['selected_text'] = self.train_val_df['selected_text'].astype(str)

        self.test_df = pd.read_csv(self.config['test_file_path'])
        self.test_df['text'] = self.test_df['text'].astype(str)