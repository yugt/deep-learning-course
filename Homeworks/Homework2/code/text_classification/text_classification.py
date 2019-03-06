import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import os
import time
import copy
import torchtext
from collections import OrderedDict


class TextDataset(Dataset):
    unk = "\a"
    padding = "\b"
    max_seq_len = 16

    def __init__(self, file_path) -> None:
        super(TextDataset, self).__init__()
        self.text_data = self.read_data(file_path)
        self.data = self.text_data
        self.vocab = None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def read_data(file):
        data = []
        with open(file, "r") as file_hdl:
            curr_line = file_hdl.readline().strip()
            while curr_line:
                label, *sentence = curr_line.split()
                data.append((sentence, int(label)))
                curr_line = file_hdl.readline().strip()
        return data

    def create_vocab(self):
        vocab = OrderedDict([(self.padding, 0), (self.unk, 1)])
        word_idx = 2
        for item in self.text_data:
            for word in item[0]:
                if word not in vocab:
                    vocab[word] = word_idx
                    word_idx += 1
        self.vocab = vocab

    def encode(self, vocab=None):
        if vocab is None:
            vocab = self.vocab
        self.data = []
        for item in self.text_data:
            sentence, label = item
            inds = []
            for word in sentence:
                inds.append(vocab.get(word, vocab[self.unk]))

            while len(inds) < self.max_seq_len:
                inds.append(vocab[self.padding])
            self.data.append([np.array(inds[:self.max_seq_len]), label])


class UnlabelledDataset(TextDataset):

    @staticmethod
    def read_data(file):
        data = []
        with open(file, "r") as file_hdl:
            curr_line = file_hdl.readline().strip()
            while curr_line:
                sentence = curr_line.split()
                data.append((sentence, 0))
                curr_line = file_hdl.readline().strip()
        return data


def train_model(dev, data_loaders, dataset_sz, model, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(dev)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # Training for num_epochs steps
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(dev)
                labels = labels.to(dev)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = torch.squeeze(outputs)
                    preds = (outputs > 0).long()
                    loss = criterion(outputs, labels.float())
                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sz[phase]
            epoch_acc = running_corrects.double() / dataset_sz[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best dev Acc: {:4f}'.format(best_acc))

    # load best model weights
    if num_epochs > 0:
        model.load_state_dict(best_model_wts)
    return model


def eval_model(dev, data_loaders, dataset_sz, model):
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        running_corrects = 0
        for inputs, labels in data_loaders["test"]:
            inputs = inputs.to(dev)
            labels = labels.to(dev)

            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            preds = (outputs > 0).long()
            running_corrects += torch.sum(preds == labels)

        acc = running_corrects.double() / dataset_sz["test"]
    return acc


def predict(dev, data_loader, model, out_file):
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        with open(out_file, "w") as file_hdl:
            for inputs, labels in data_loader:
                inputs = inputs.to(dev)
                outputs = model(inputs)
                outputs = torch.squeeze(outputs)
                preds = outputs > 0
                file_hdl.write(str(preds.item()) + "\n")

