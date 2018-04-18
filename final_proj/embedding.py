from abc import ABCMeta, abstractmethod
import numpy as np
# from gensim.models import KeyedVectors

def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (vocabulary.size, vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.word_to_id.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors

class Embedding(metaclass=ABCMeta):
    @abstractmethod
    def get_w(self):
        pass

    @abstractmethod
    def is_trainable(self):
        pass


class RandomEmbedding(Embedding):
    def __init__(self, vocab_size, embedding_size, init_scale=0.001):
        self.rows = vocab_size
        self.columns = embedding_size
        self.train_embeddings = True
        self.w = np.random.uniform(-init_scale, init_scale, size=((self.rows, self.columns))).astype("float32")

    def get_w(self):
        return self.w

    def is_trainable(self):
        return self.train_embeddings


class Word2VecEmbedding(Embedding):
    def __init__(self, embedding_path, vocab, is_trainable):
        self.embedding_path = embedding_path
        self.vocab = vocab
        self.train_embeddings = is_trainable
        self.rows = vocab.size

        w2v = load_embedding_vectors_word2vec(vocab,embedding_path, binary=True)
        self.columns = 300
        # self.w = np.random.uniform(-1.0, 1.0, size=((self.rows, self.columns))).astype("float32")
        self.w = w2v

    def get_w(self):
        return self.w

    def is_trainable(self):
        return self.train_embeddings
    
