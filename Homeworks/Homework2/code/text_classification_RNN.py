from text_classification_GloVe import *

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
