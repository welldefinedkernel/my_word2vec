import urllib.request
from text_preprocessor import TextPreprocessor
from word2vec import Word2Vec

def load_dataset():
    # Tiny Shakespeare dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print("Downloading Tiny Shakespeare dataset...")
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')
    print(f"Dataset downloaded. Total characters: {len(data)}")
    
    return data[:20000]

def main():
    raw_text = load_dataset()

    print("Preprocessing text...")
    txt_preprocessor = TextPreprocessor()

    word2vec = Word2Vec(txt_preprocessor, embedding_dim=100, negative_samples=10, seed=42)
    
    print("Starting training...")
    word2vec.train(raw_text, epochs=20)
    
    test_word = "citizen"
    if test_word in txt_preprocessor.word2idx:
        idx = txt_preprocessor.word2idx[test_word]
        emb = word2vec.get_embedding(idx)
        print(f"\nEmbedding for '{test_word}':")
        print(emb)
    else:
        print(f"\nWord '{test_word}' not found in vocabulary.")

if __name__ == "__main__":
    main()
