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

        pos_pred = self._sigmoid(np.dot(e_w, v_ci))                         # Prediction for positive sample
        neg_preds = [(self._sigmoid(np.dot(e_w, v_cj))) for  v_cj in v_cjs] # Predictions for negative samples

        return {"emb" : e_w, 
                "pos" : v_ci, 
                "neg" : v_cjs,
                "pos_red" : pos_pred,
                "neg_preds" : neg_preds}

    def backward(self, cache):
        e_w = cache["emb"]
        v_ci = cache["pos"]
        v_cjs = cache["neg"]

        pos_pred = cache["pos_pred"]
        neg_preds = cache["neg_preds"]

        e_w_grad = (pos_pred - 1) * v_ci + np.sum([neg_pred * v_cj for neg_pred, v_cj in zip(neg_preds, v_cjs)])
        v_ci_grad = (pos_pred - 1) * e_w 
        v_cj_grads = [neg_pred * e_w for neg_pred in neg_preds]

        return {"emb_gr" : e_w_grad, "pos_gr" : v_ci_grad, "neg_gr" : v_cj_grads}

    def train(self, corpus: str, epochs=3):
        corpus = corpus.lower()
        corpus = re.sub(r'[^a-z\s]', '', corpus)

        tokens = ["_"] * self.C + corpus.split() + ["_"] * self.C
        print(tokens)

        for _ in range(epochs):
            for i, center in enumerate(tokens[self.C : -self.C]):
                R = np.random.randint(1, self.C)
                for j, context in enumerate(tokens[i - self.R : i + self.R]):
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
