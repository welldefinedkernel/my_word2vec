from text_preprocessor import TextPreprocessor
from word2vec import Word2Vec

def main():
    raw_text = "Hello, my name is Roman! This is a simple test corpus for Word2Vec training!"
    txt_preprocessor = TextPreprocessor()

    word2vec = Word2Vec(txt_preprocessor, embedding_dim=10, negative_samples=1)
    word2vec.train(raw_text)


if __name__ == "__main__":
    main()
