import re
from collections import Counter

class TextPreprocessor:
    """
    Class to handle text preprocessing, vocabulary creating and storing
    statistics for training Word2Vec.
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

        self.word_counts = Counter()
        self.vocab_size = 0
        
        self.vocab_size = 0
        self.word_freqs = {}
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()

    def build_vocab(self, tokens):
        counts = Counter(tokens)
        self.word_counts = Counter({w: c for w, c in counts.items()})
        
        for word in self.word_counts:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
        self.vocab_size = len(self.word2idx)
        
        if self.vocab_size > 0:
            total_words = sum(self.word_counts.values())
            self.word_freqs = {word: count / total_words for word, count in self.word_counts.items()}
        
    def text_to_indices(self, tokens):
        return [self.word2idx[w] for w in tokens if w in self.word2idx]

    def process(self, text):
        tokens = self.preprocess_text(text)
        self.build_vocab(tokens)
        return self.text_to_indices(tokens)
