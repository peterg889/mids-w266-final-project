from w266_common import utils, vocabulary
from sentence import Sentence
import math, random
import numpy as np
# from nltk.tokenize.treebank import TreebankWordTokenizer

KID_INAPPROPRIATE = 0
KID_APPROPRIATE = 1

def split_frac(x, frac):
    split_index = math.floor(len(x) * frac)
    return x[:split_index], x[split_index:]

def shuffle(x):
    x = list(x)
    random.shuffle(x)
    return x

def extend_y(y):
    ret = []
    for row in y:
        if row == KID_APPROPRIATE:
            ret.append([0, 1])
        else:
            ret.append([1, 0])
    return np.array(ret)

class TrainingSet:
    def __init__(self, pos_files = None, neg_files = None, tokenizer = None, max_examples = 0, name=""):
        if pos_files:
            self.pos_sentences = self._load_example_set(pos_files, KID_APPROPRIATE, tokenizer,  max_examples)
        else:
            self.pos_sentences = None
        if neg_files:
            self.neg_sentences = self._load_example_set(neg_files, KID_INAPPROPRIATE,tokenizer, max_examples)
        else:
            self.neg_sentences = None
        # self.tokenizer = TreebankWordTokenizer()
        self.word_list = None
        self.vocab = None
        self.sents_prepped = False
        self.sds = None
        self.trains = None
        self.devs = None
        self.sd_test_sents = None
        self.train_sents = None
        self.dev_sents = None
        self.test_sentences = None
        self.name = name
        
    def load_test_arr(self, test_arr, tokenizer):
        self.test_sentences = self._load_test_arr(test_arr, tokenizer)


    def _generate_word_list(self):
        all_words = []
        all_sents = self.pos_sentences + self.neg_sentences
        for sent in all_sents:
            for word in sent.tokens:
                all_words.append(word)
        self.word_list = all_words
            
        
    def generate_vocab(self, max_size=None):
        self._generate_word_list()
        self.vocab = vocabulary.Vocabulary(self.word_list, size=max_size)
    
    def set_vocab(self, vocab):
        self.vocab = vocab
        
    def _load_example_set(self, file_names, label, tokenizer, max_examples = 0):
        examples = []
        i = 0
        for n in file_names:
            with open(n, "r") as f:
                for line in f:
                    if max_examples != 0 and i == max_examples:
                        break
                    tokens = tokenizer.tokenize(line)
                    canonical_tokens = utils.canonicalize_words(tokens)
                    sent = Sentence(line, label, canonical_tokens)
                    examples.append(sent)
                    i += 1
        return examples
    
    def _load_test_arr(self, test_arr, tokenizer):
        test_sents = []
        for row in test_arr:
            tokens = tokenizer.tokenize(row[0])
            canonical_tokens = utils.canonicalize_words(tokens)
            sent = Sentence(row[0], row[1], canonical_tokens)
            test_sents.append(sent)
        return test_sents
    
    def prep_sents(self, pad_len):
        if self.pos_sentences:
            for sent in self.pos_sentences:
                sent.pad_to(pad_len, self.vocab.PAD_TOKEN)
                sent.apply_vocabs(self.vocab) 
        if self.neg_sentences:
            for sent in self.neg_sentences:
                sent.pad_to(pad_len, self.vocab.PAD_TOKEN)
                sent.apply_vocabs(self.vocab) 
        if self.test_sentences:
            for sent in self.test_sentences:
                sent.pad_to(pad_len, self.vocab.PAD_TOKEN)
                sent.apply_vocabs(self.vocab) 
        self.sents_prepped = True

    def prep_sets(self):
        assert self.sents_prepped
        
        if self.sd_test_sents:
            x, ns, y = [],[],[]
            for sent in self.sd_test_sents:
                assert sent.vocab_applied and sent.padded
                x.append(sent.padded_tokens_ids)
                ns.append(len(sent.tokens))
                y.append(sent.label)
            x_np = np.array(x)
            ns_np = np.array(ns)
            y_np = np.array(y)
            self.sds = (x_np,ns_np,y_np, extend_y(y_np))
            
        if self.train_sents:
            x, ns, y = [],[],[]
            for sent in self.train_sents:
                assert sent.vocab_applied and sent.padded
                x.append(sent.padded_tokens_ids)
                ns.append(len(sent.tokens))
                y.append(sent.label)
            x_np = np.array(x)
            ns_np = np.array(ns)
            y_np = np.array(y)
            self.trains = (x_np,ns_np,y_np, extend_y(y_np))          
        if self.dev_sents:
            x, ns, y = [],[],[]
            for sent in self.dev_sents:
                assert sent.vocab_applied and sent.padded
                x.append(sent.padded_tokens_ids)
                ns.append(len(sent.tokens))
                y.append(sent.label)
            x_np = np.array(x)
            ns_np = np.array(ns)
            y_np = np.array(y)
            self.devs = (x_np,ns_np,y_np, extend_y(y_np))
        if self.test_sentences:
            x, ns, y = [],[],[]
            for sent in self.test_sentences:
                assert sent.vocab_applied and sent.padded
                x.append(sent.padded_tokens_ids)
                ns.append(len(sent.tokens))
                y.append(sent.label)
            x_np = np.array(x)
            ns_np = np.array(ns)
            y_np = np.array(y)
            self.ext_tests = (x_np,ns_np,y_np, extend_y(y_np))   
    
    def divide_test_train(self, test_frac, train_frac, randomize = True):
        all_sents = self.pos_sentences + self.neg_sentences
        if randomize:
            all_sents = shuffle(all_sents)
        self.sd_test_sents, usable_sents = split_frac(all_sents, test_frac)
        self.train_sents, self.dev_sents = split_frac(usable_sents, train_frac)
    
