from w266_common import utils, vocabulary

class Sentence:
    def __init__(self, sentence, label, tokens):
        self.sentence = sentence
        self.label = label
        self.tokens = tokens
        self.token_ids = []
        self.padded_tokens = []
        self.padded_tokens_ids = []

        self.padded = False
        self.vocab_applied = False
        # self.labels_ids = []

    def pad_to(self, length, pad_token):
        diff = length - len(self.tokens)
        if diff >= 0:
            new_tokens = [pad_token] * diff
            self.padded_tokens = self.tokens + new_tokens
        else:
            self.padded_tokens = self.tokens[:diff]
        self.padded = True
        

    def apply_vocabs(self, vocab):
        if self.padded_tokens:
            self.padded_tokens_ids = vocab.words_to_ids(self.padded_tokens)
        self.token_ids = vocab.words_to_ids(self.tokens)
        self.vocab_applied = True
