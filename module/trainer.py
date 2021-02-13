import numpy as np
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from module.model import LinearNN


class Loss_Function(object):
    # Loss Function 1: Cross-entropy
    @staticmethod
    def loss_fn(start_logits, end_logits, start_positions, end_positions):
        ce_loss = nn.CrossEntropyLoss()
        start_loss = ce_loss(start_logits, start_positions)
        end_loss = ce_loss(end_logits, end_positions)
        total_loss = start_loss + end_loss
        return total_loss

    # Loss Function 2: Distance
    @staticmethod
    def dist_loss(start_logits, end_logits, start_positions, end_positions, max_len, device='cuda', scale=1):
        """calculate distance loss between prediction's length & GT's length

        Input
        - start_logits ; shape (batch, max_seq_len{128})
            - logits for start index
        - end_logits
            - logits for end index
        - start_positions ; shape (batch, 1)
            - start index for GT
        - end_positions
            - end index for GT
        """
        start_logits = torch.nn.Softmax(1)(start_logits)  # shape ; (batch, max_seq_len)
        end_logits = torch.nn.Softmax(1)(end_logits)

        start_one_hot = torch.nn.functional.one_hot(start_positions, num_classes=max_len).to(device)
        end_one_hot = torch.nn.functional.one_hot(end_positions, num_classes=max_len).to(device)

        pred_dist = Loss_Function.__dist_between(start_logits, end_logits, device, max_len)
        gt_dist = Loss_Function.__dist_between(start_one_hot, end_one_hot, device, max_len)  # always positive
        diff = (gt_dist - pred_dist)

        rev_diff_squared = 1 - torch.sqrt(diff * diff)  # as diff is smaller, make it get closer to the one
        loss = -torch.log(
            rev_diff_squared)  # by using negative log function, if argument is near zero -> inifinite, near one -> zero

        return loss * scale

    @staticmethod
    def __dist_between(start_logits, end_logits, max_len, device='cuda'):
        """get dist btw. pred & ground_truth"""

        linear_func = torch.tensor(np.linspace(0, 1, max_len, endpoint=False), requires_grad=False)
        linear_func = linear_func.to(device)

        start_pos = (start_logits * linear_func).sum(axis=1)
        end_pos = (end_logits * linear_func).sum(axis=1)

        diff = end_pos - start_pos

        return diff.sum(axis=0) / diff.size(0)


class Evaluation_Function(object):
    @staticmethod
    def jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
        start_pred = np.argmax(start_logits)
        end_pred = np.argmax(end_logits)
        if start_pred > end_pred:
            pred = text
        else:
            pred = Evaluation_Function.__get_selected_text(text, start_pred, end_pred, offsets)

        true = Evaluation_Function.__get_selected_text(text, start_idx, end_idx, offsets)

        return Evaluation_Function.__compute_jaccard_value(true, pred)

    @staticmethod
    def __get_selected_text(text, start_idx, end_idx, offsets):
        selected_text = ""
        for ix in range(start_idx, end_idx + 1):
            selected_text += text[offsets[ix][0]: offsets[ix][1]]
            if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                selected_text += " "
        return selected_text

    @staticmethod
    def __compute_jaccard_value(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))


class Trainer(object):
    def __init__(self, config, logger, preprocessor):
        self.config, self.logger = config, logger
        self.preprocessor = preprocessor
        self.model = LinearNN(self.config)
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.config['learning_rate'], betas=(0.9, 0.999), weight_decay=1.2e-2)

    def fit(self):
        train_val_loaders_dict = {"train": self.preprocessor.train_loader, "val": self.preprocessor.val_loader}
        epochs = self.config['epochs']

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

        for epoch in range(epochs):
            for phase in ['train', 'val']:
                # Time
                start_time = time.time()

                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                # Initialize Loss and Jaccard
                epoch_loss, epoch_jaccard = 0.0, 0.0

                for data in tqdm(train_val_loaders_dict[phase]):
                    ids, masks, tweet, offsets, start_idx, end_idx = Trainer.__unpack_data(data)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # Propagate Forward
                        start_logits, end_logits = self.model(ids, masks)

                        # Calculate Loss and Propagate Backward
                        loss = Loss_Function.loss_fn(start_logits, end_logits, start_idx, end_idx)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            # scheduler.step()
                        epoch_loss += loss.item() * len(ids)

                        # Get Start/End Index/Logits
                        start_idx = start_idx.cpu().detach().numpy()
                        end_idx = end_idx.cpu().detach().numpy()
                        start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                        end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

                        # Calculate Jaccard
                        for i in range(len(ids)):
                            jaccard_score = Evaluation_Function.jaccard_score(
                                tweet[i], start_idx[i], end_idx[i], start_logits[i], end_logits[i], offsets[i])
                            epoch_jaccard += jaccard_score

                # Average Loss and Jaccard Score
                epoch_loss = epoch_loss / len(train_val_loaders_dict[phase].dataset)
                epoch_jaccard = epoch_jaccard / len(train_val_loaders_dict[phase].dataset)

                # Time
                end_time = time.time()
                cost = end_time - start_time

                # Print Summary for Each Epoch
                print(f'Epoch {epoch + 1}/{epochs} {cost:.1f}s | {phase:^5} | Loss: {epoch_loss:.4f} | Jaccard: {epoch_jaccard:.4f}')

                # Only Save Model with Better Jaccard
                # if phase == 'val' and epoch_jaccard > initial_jaccard:
                #     print('saving...')
                #     initial_jaccard = epoch_jaccard
                #     torch.save(self.model.state_dict(), filename)

    @staticmethod
    def __unpack_data(batch_data):
        if torch.cuda.is_available():
            ids = batch_data['ids'].cuda()
            masks = batch_data['masks'].cuda()
            tweet = batch_data['tweet']
            offsets = batch_data['offsets'].numpy()
            start_idx = batch_data['start_idx'].cuda()
            end_idx = batch_data['end_idx'].cuda()
        else:
            ids = batch_data['ids'].cpu()
            masks = batch_data['masks'].cpu()
            tweet = batch_data['tweet']
            offsets = batch_data['offsets'].numpy()
            start_idx = batch_data['start_idx'].cpu()
            end_idx = batch_data['end_idx'].cpu()
        return ids, masks, tweet, offsets, start_idx, end_idx