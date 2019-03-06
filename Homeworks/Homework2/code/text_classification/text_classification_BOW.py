from text_classification import *


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


