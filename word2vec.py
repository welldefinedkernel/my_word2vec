import numpy as np
from text_preprocessor import TextPreprocessor

class Word2Vec:
    """
    Word2Vec model. 
    Trained using Skip-gram with Negative Sampling.
    """

    def __init__(self, 
                 preprocessed_text : TextPreprocessor,
                 embedding_dim : int, 
                 learning_rate=1e-3, 
                 window_size = 5,
                 negative_samples=5):

        self.corpus = preprocessed_text

        self.v = preprocessed_text.vocab_size
        self.d = embedding_dim

        self.E = np.ones((self.v, self.d)) # Input embeddings
        self.W = np.ones((self.v, self.d)) # Output embeddings

        self.alpha = learning_rate
        self.k = negative_samples

        self.C = window_size # Max amount of words from context to train on

    def _sigmoid(self, x):
        return np.reciprocal(1 + np.exp(-x))

    def _get_negative_samples(self, center_idx, context_idx, n_samples):
        pass

    def loss(self, cache):
        pos_pred = cache["pos_pred"]
        neg_preds = cache["neg_preds"]

        pos_loss = -np.log(pos_pred)
        neg_loss = -np.sum([np.log(1 - neg_pred) for neg_pred in neg_preds])

        L = pos_loss + neg_loss

        return L

    def forward(self, center_idx, context_idx, negative_indices):
        e_w = self.E[center_idx, :]                      # Embedding vector
        v_ci = self.W[context_idx, :]                    # Positive sample 
        v_cjs = [self.W[i, :] for i in negative_indices] # Negative samples

        pos_pred = self._sigmoid(np.dot(e_w, v_ci))                         # Prediction for positive sample
        neg_preds = [(self._sigmoid(np.dot(e_w, v_cj))) for v_cj in v_cjs]  # Predictions for negative samples

        return {"emb" : e_w, 
                "pos" : v_ci, 
                "neg" : v_cjs,
                "pos_pred" : pos_pred,
                "neg_preds" : neg_preds}

    def backward(self, cache):
        e_w = cache["emb"]
        v_ci = cache["pos"]
        v_cjs = cache["neg"]

        pos_pred = cache["pos_pred"]
        neg_preds = cache["neg_preds"]

        e_w_grad = (pos_pred - 1) * v_ci
        e_w_grad += np.sum([neg_pred * v_cj for neg_pred, v_cj in zip(neg_preds, v_cjs)], axis=0)
        v_ci_grad = (pos_pred - 1) * e_w 
        v_cj_grads = [neg_pred * e_w for neg_pred in neg_preds]

        return {"emb_gr" : e_w_grad, 
                "pos_gr" : v_ci_grad, 
                "neg_gr" : v_cj_grads}

    def train(self, epochs=3):
        for _ in epochs:
            ...

    def get_embedding(self, word_idx):
        return self.E[word_idx, :] 

    def get_embedding_matrix(self):
        return self.E

def main():
    raw_text = "Hello, my name is Roman!"
    txt_preprocessor = TextPreprocessor()
    txt_preprocessor.process(raw_text)

    word2vec = Word2Vec(txt_preprocessor, embedding_dim=10)
    word2vec.train()

if __name__ == "__main__":
    main()
