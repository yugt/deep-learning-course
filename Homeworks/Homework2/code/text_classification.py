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


class BOWClassifier(nn.Module):
    def __init__(self, vocab_size):
        super(BOWClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.fc = nn.Linear(in_features=vocab_size, out_features=1)

    def forward(self, x):
        batch_sz = x.shape[0]
        embeds = torch.zeros((batch_sz, self.vocab_size))
        embeds[torch.arange(batch_sz), x.t()] = 1
        fc_out = self.fc(embeds)
        return fc_out


class AvgPoolClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim=50, pad_idx=0):
        super(AvgPoolClassifier, self).__init__()
        self.embed = nn.Embedding(embedding_dim=embed_dim,
                                  num_embeddings=vocab_size,
                                  padding_idx=pad_idx)
        self.fc = nn.Linear(in_features=embed_dim, out_features=1)

    def forward(self, x):
        num_non_zeros = (x != 0).sum(dim=1, keepdim=True).float()
        embed_out = self.embed(x)
        embed_out = embed_out.sum(dim=1) / num_non_zeros
        fc_out = self.fc(embed_out)
        return fc_out


class GloVeClassifier(AvgPoolClassifier):

    def __init__(self, vocab, embed_dim=50, pad_idx=0, unk_idx=1):
        super().__init__(len(vocab), embed_dim, pad_idx)
        glove = torchtext.vocab.GloVe(name='6B', dim=embed_dim)
        glove_weights = []
        for word in vocab.keys():
            glove_weights.append(glove.vectors[glove.stoi.get(word, 0)])
        glove_weights[unk_idx] = torch.ones_like(glove_weights[unk_idx])
        self.embed = nn.Embedding.from_pretrained(torch.stack(glove_weights),
                                                  freeze=False)
        self.embed.padding_idx = pad_idx


class RNNClassifier(GloVeClassifier):

    def __init__(self, vocab, embed_dim=50, rnn_num_hid=64,
                 rnn_bidir=False, pad_idx=0, unk_idx=1):
        super().__init__(vocab, embed_dim, pad_idx, unk_idx)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=rnn_num_hid,
                          batch_first=True, bidirectional=rnn_bidir)
        self.fc = nn.Linear(in_features=(rnn_bidir + 1) * rnn_num_hid,
                            out_features=1)

    def forward(self, x):
        embed_out = self.embed(x)
        _, rnn_out = self.rnn(embed_out)
        if isinstance(rnn_out, tuple):
            rnn_out = rnn_out[0]
        fc_out = self.fc(rnn_out.view(rnn_out.shape[1], -1))
        return fc_out


class LSTMClassifier(RNNClassifier):

    def __init__(self, vocab, embed_dim=50, rnn_num_hid=64,
                 rnn_bidir=False, pad_idx=0, unk_idx=1):
        super(LSTMClassifier, self).__init__(vocab, embed_dim, rnn_num_hid,
                                             rnn_bidir, pad_idx, unk_idx)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=rnn_num_hid,
                           batch_first=True, bidirectional=rnn_bidir)


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


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    data_dir = "data"
    text_datasets = {x: TextDataset(os.path.join(data_dir, x + ".txt"))
                     for x in ["train", "dev", "test"]}

    text_datasets["train"].create_vocab()
    text_vocab = text_datasets["train"].vocab

    for x in ["train", "dev", "test"]:
        text_datasets[x].encode(text_vocab)

    dataloaders = {x: DataLoader(text_datasets[x], batch_size=32, shuffle=True,
                                 num_workers=4)
                   for x in ["train", "dev", "test"]}
    dataset_sizes = {x: len(text_datasets[x]) for x in ["train", "dev", "test"]}

    unlabelled_dataset = UnlabelledDataset(os.path.join(data_dir, "unlabelled.txt"))
    unlabelled_dataset.encode(text_vocab)
    unlbl_dataloader = DataLoader(unlabelled_dataset, batch_size=1)

    models = {"BOW": BOWClassifier(vocab_size=len(text_vocab)),
              "Avg_Pool": AvgPoolClassifier(vocab_size=len(text_vocab)),
              "GloVe": GloVeClassifier(vocab=text_vocab),
              "RNN": RNNClassifier(vocab=text_vocab, embed_dim=50, rnn_num_hid=128),
              "LSTM": LSTMClassifier(vocab=text_vocab, embed_dim=50, rnn_num_hid=128)}

    count = 1
    for model_nm in models.keys():
        models[model_nm] = train_model(device, dataloaders, dataset_sizes, models[model_nm])
        acc = eval_model(device, dataloaders, dataset_sizes, models[model_nm])
        print('Best test Acc for {:s} model: {:4f}'.format(model_nm, acc))
        output_file = (data_dir + "/predictions_q{:d}.txt").format(count)
        predict(device, unlbl_dataloader, models[model_nm], output_file)
        count += 1