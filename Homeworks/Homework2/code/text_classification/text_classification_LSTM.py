from text_classification_RNN import *

class LSTMClassifier(RNNClassifier):

    def __init__(self, vocab, embed_dim=50, rnn_num_hid=64,
                 rnn_bidir=False, pad_idx=0, unk_idx=1):
        super(LSTMClassifier, self).__init__(vocab, embed_dim, rnn_num_hid,
                                             rnn_bidir, pad_idx, unk_idx)
        self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=rnn_num_hid,
                           batch_first=True, bidirectional=rnn_bidir)
