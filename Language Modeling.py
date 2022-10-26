import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    start = []
    for i in range(1,n):
        start.append("<s>")
    text = start + text + ['</s>']
    ngrams = []
    for i in range(n-1,len(text)):
        context = []
        for j in range(i-(n-1),i):
            context.append(text[j])
        ngrams.append((text[i],tuple(context)))
    return ngrams

# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    f = open(corpus_path, 'r')
    text = f.read()
    f.close()
    para = text.split('\n\n')
    lines = []
    for p in para:
        lines+=sent_tokenize(p)
    text = []
    for i in lines:
        text.append(word_tokenize(i))
    return text

# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    text = load_corpus(corpus_path)
    LM = NGramLM(n)
    for i in text:
        LM.update(i)
    return LM

# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        ngrams = get_ngrams(self.n, text)
        for i in ngrams:
            if i in self.ngram_counts:
                self.ngram_counts[i]+=1
            else:
                self.ngram_counts[i]=1
            if i[1] in self.context_counts:
                self.context_counts[i[1]]+=1
            else:
                self.context_counts[i[1]]=1
            if i[0] not in self.vocabulary:
                self.vocabulary.add(i[0])


    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        if context not in self.context_counts:
            return 1/len(self.vocabulary)
        if delta == 0:
            if (word, context) not in self.ngram_counts:
                return 0
            return self.ngram_counts[(word, context)]/self.context_counts[context]
        else:
            if (word, context) not in self.ngram_counts:
                return (delta)/(self.context_counts[context] + (delta*len(self.vocabulary)))
            return (self.ngram_counts[(word, context)] + delta)/(self.context_counts[context] + (delta*len(self.vocabulary)))

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        ngrams = get_ngrams(self.n, sent)
        sent_prob = 0
        #print(ngrams,'ngrams')
        for i in range(len(ngrams)-1):
            #print(ngrams[i][1],self.context_counts[ngrams[i][1]],'cc')
            prob = self.get_ngram_prob(ngrams[i][0],ngrams[i][1],delta)
            #print(prob,'prob')
            if prob == 0:
                return -math.inf
            log_prob = math.log(prob, 2)
            sent_prob+=log_prob
        return sent_prob

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]]) -> float:
        total_log_prob = 0
        total_tokens = 0
        for i in corpus:
            for j in i:
                sent_prob = self.get_sent_log_prob(word_tokenize(j))
                total_log_prob+=sent_prob
                total_tokens+=len(word_tokenize(j))
        #print('tt',total_log_prob, total_tokens)
        perplexity = math.pow(2,-(total_log_prob/total_tokens))
        return perplexity

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        vocab = list(self.vocabulary)
        vocab = sorted(vocab)
        r = random.random()
        prob_list = []
        for i in vocab:
            prob = self.get_ngram_prob(i, context, delta)
            prob_list.append(prob)
        curr = 0
        for i in range(len(prob_list)):
            curr+=prob_list[i]
            if r<curr:
                return vocab[i]

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        start = []
        for i in range(1,self.n):
            start.append("<s>")
        sentence = []
        first = self.generate_random_word(tuple(start))
        sentence.append(first)
        for i in range(max_length-1):
            nxt = self.generate_random_word(tuple(start[i+1:]+sentence[len(sentence)-(self.n-1):]))
            if nxt == '</s>':
                sentence.append(nxt)
                break
            sentence.append(nxt)
        return ' '.join(sentence)


def main(corpus_path: str, delta: float, seed: int):
    trigram_lm = create_ngram_lm(2, corpus_path)
    s1 = 'God has given it to me, let him who touches it beware!'
    s2 = 'Where is the prince, my Dauphin?'

    #print(trigram_lm.get_sent_log_prob(word_tokenize(s1), delta))
    #print(trigram_lm.get_sent_log_prob(word_tokenize(s2), delta))  
    corpus = [[s1],[s1]]
    print(trigram_lm.get_perplexity(corpus))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str, default='warpeace.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=0.0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
