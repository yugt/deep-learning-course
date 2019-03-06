from text_classification import *

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
