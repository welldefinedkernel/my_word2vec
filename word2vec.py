import numpy as np
import re

class Word2Vec:
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 learning_rate=1e-3, 
                 window_size = 5,
                 negative_samples=5):

        self.E = np.ones((vocab_size, embedding_dim)) # Input embeddings
        self.W = np.ones((vocab_size, embedding_dim)) # Output embeddings
        
        self.v = vocab_size
        self.d = embedding_dim

        self.alpha = learning_rate
        self.k = negative_samples

        self.C = window_size # Max amount of words from context to train on

    def _sigmoid(self, x):
        return np.reciprocal(1 + np.exp(-x))

    def _get_negative_samples(self, center_idx, context_idx, n_samples):
        pass

    def loss(self, cache):
        e_w = cache["emb"]
        v_ci = cache["pos"]
        v_cjs = cache["neg"]

        pos_loss = -np.log(self._sigmoid(np.dot(e_w, v_ci)))
        neg_loss = -np.sum([np.log(1 - self._sigmoid(np.dot(e_w, v_cj))) for v_cj in v_cjs])

        L = pos_loss + neg_loss

        return L

    def forward(self, center_idx, context_idx, negative_indices):
        e_w = self.E[center_idx, :]                      # Embedding vector
        v_ci = self.W[context_idx, :]                    # Positive sample 
        v_cjs = [self.W[i, :] for i in negative_indices] # Negative samples

        return {"emb" : e_w, "pos" : v_ci, "neg" : v_cjs}

    def backward(self, cache):
        e_w = cache["emb"]
        v_ci = cache["pos"]
        v_cjs = cache["neg"]

        e_w_grad = (self._sigmoid(np.dot(e_w, v_ci)) - 1) * v_ci + np.sum([(self._sigmoid(np.dot(e_w, v_cj))) * v_cj for v_cj in v_cjs])
        v_ci_grad = (self._sigmoid(np.dot(e_w, v_ci)) - 1) * e_w 
        v_cj_grads = [(self._sigmoid(np.dot(e_w, v_cj))) * e_w for v_cj in v_cjs]

        return {"emb_gr" : e_w_grad, "pos_gr" : v_ci_grad, "neg_gr" : v_cj_grads}

    def train(self, corpus: str, epochs=3):
        ...

    def get_embedding(self, word_idx):
        return self.E[word_idx, :] 

    def get_embedding_matrix(self):
        return self.E

def main():
    word2vec = Word2Vec(10, 10)
    cache = word2vec.forward(5, 5, [0, 1, 2, 3, 4])
    print(word2vec.loss(cache))
    word2vec.train("Hello, my name is Roman!")

if __name__ == "__main__":
    main()
