from text_classification_AvgPool import *

class GloVeClassifier(AvgPoolClassifier):

    def __init__(self, vocab, embed_dim=50, pad_idx=0, unk_idx=1):
        super().__init__(len(vocab), embed_dim, pad_idx)
        glove = torchtext.vocab.GloVe(name='6B', dim=embed_dim)
        glove_weights = []
        for word in vocab.keys():
            glove_weights.append(glove.vectors[glove.stoi.get(word, 0)])
        glove_weights[unk_idx] = torch.ones_like(glove_weights[unk_idx])
        self.embed = nn.Embedding.from_pretrained(torch.stack(glove_weights), freeze=False)
        self.embed.padding_idx = pad_idx

