import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim, learning_rate=1e-3, negative_samples=5):
        self.E = np.ones((vocab_size, embedding_dim)) # Input embeddings
        self.W = np.ones((embedding_dim, vocab_size)) # Output embeddings
        
        self.v = vocab_size
        self.d = embedding_dim

        self.alpha = learning_rate
        self.k = negative_samples

    def _sigmoid(self, x):
        pass

    def _get_negative_samples(self, context_idx, n_samples):
        pass

    def forward(self, center_idx, context_idx, negative_indices):
        pass

    def backward(self, center_idx, context_idx, negative_indices, cache):
        pass

    def train(self, corpus, epochs):
        pass

    def get_embedding(self, word_idx):
        pass

def main():
    pass

if __name__ == "__main__":
    main()
