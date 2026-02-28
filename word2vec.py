import numpy as np
from text_preprocessor import TextPreprocessor

class Word2Vec:
    """
    Word2Vec model. 
    Trained using Skip-gram with Negative Sampling.
    """

    def __init__(self, 
                 preprocessor : TextPreprocessor,
                 embedding_dim : int, 
                 learning_rate=1e-3, 
                 window_size=5,
                 negative_samples=5,
                 seed=None):

        self.preprocessor = preprocessor
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.v = preprocessor.vocab_size 
        self.d = embedding_dim

        self.E = None # Input embeddings
        self.W = None # Output embeddings

        self.alpha = learning_rate
        self.k = negative_samples

        self.C = window_size # Max amount of words from context to train on

    def _sigmoid(self, x):
        return np.reciprocal(1 + np.exp(-x))

    def _get_negative_samples(self, center_idx, context_idx, n_samples):
        exclude = [center_idx, context_idx]
        all_indices = np.arange(self.v)
        possible_indices = np.setdiff1d(all_indices, exclude)

        return np.random.choice(possible_indices, size=n_samples, replace=False)


    def loss(self, cache):
        pos_pred = cache["pos_pred"]
        neg_preds = cache["neg_preds"]

        eps = 1e-10
        pos_loss = -np.log(pos_pred + eps)
        neg_loss = -np.sum(np.log(1 - neg_preds + eps))

        L = np.sum(pos_loss + neg_loss)

        return L

    def forward(self, center_idx, context_idx, negative_indices):
        e_w = self.E[center_idx, :]         # Embedding vector
        v_ci = self.W[context_idx, :]       # Positive sample 
        v_cjs = self.W[negative_indices, :] # Negative samples

        pos_pred = self._sigmoid(np.dot(e_w, v_ci)) # Prediction for positive sample
        neg_preds = self._sigmoid(v_cjs @ e_w.T)    # Predictions for negative samples

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


        e_w_grad = (pos_pred - 1) * v_ci + np.sum(neg_preds.reshape(-1, 1) * v_cjs)
        v_ci_grad = (pos_pred - 1) * e_w 
        v_cj_grads = np.outer(neg_preds, e_w)

        return {"emb_gr" : e_w_grad, 
                "pos_gr" : v_ci_grad, 
                "neg_gr" : v_cj_grads}

    def train(self, corpus: str, epochs=3):
        tokens = self.preprocessor.process(corpus)
        
        # Retrieve vocab size and initialize embedding weights
        self.v = self.preprocessor.vocab_size
        
        limit = np.sqrt(6 / (self.v + self.d))
        self.E = np.random.uniform(-limit, limit, (self.v, self.d)) 
        self.W = np.random.uniform(-limit, limit, (self.v, self.d))

        for epoch in range(epochs):
            total_loss = 0
            for i, center_word_idx in enumerate(tokens):
        
                # Dynamic window size for center word
                current_window_size = np.random.randint(1, self.C + 1)
                start = max(0, i - current_window_size)
                end = min(len(tokens), i + current_window_size + 1)
                
                for j in range(start, end):
                    if i == j: 
                        continue # Skip center word
                    
                    context_word_idx = tokens[j]
                    
                    negative_samples = self._get_negative_samples(center_word_idx, context_word_idx, self.k)
                    
                    cache = self.forward(center_word_idx, context_word_idx, negative_samples)
                    
                    loss = self.loss(cache)
                    total_loss += loss
                    
                    grads = self.backward(cache)
                    
                    self.E[center_word_idx, :] -= self.alpha * grads["emb_gr"]
                    self.W[context_word_idx, :] -= self.alpha * grads["pos_gr"]
                    self.W[negative_samples, :] -= self.alpha * grads["neg_gr"]
            
            print(f"Epoch: {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    def get_embedding(self, word_idx):
        return self.E[word_idx, :] 

    def get_embedding_matrix(self):
        return self.E
