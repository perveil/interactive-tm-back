import torch
from torch.utils.data import Dataset


class DocDataset(Dataset):
    def __init__(self, vocab, bow, doc_lengths, term_freqs):
        self.vocab = vocab              # 词典，['word1', 'word2', ...]
        self.vocab_size = len(vocab)
        self.bow = bow
        self.doc_lengths = doc_lengths  # 每篇文章的长度
        self.term_freqs = term_freqs    # 每个词的总出现频率

    def __getitem__(self, idx):
        bow_vec = torch.tensor(self.bow.toarray()[idx]).float()
        return bow_vec  # tensor[freq_of_word1, freq_of_word2, ...]

    def __len__(self):
        return len(self.doc_lengths)