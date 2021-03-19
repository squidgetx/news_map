# Python script to take an input corpus and output topics
from nltk.tokenize import sent_tokenize
import json
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import itertools
from gensim import corpora
from gensim import models
import gensim
from gensim.test.utils import datapath
import fileinput
import logging
import argparse
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import gensim.downloader as api
from gsdmm import MovieGroupProcess

import pdb

OTHER_STOPWORDS = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 
    'see', 'want', 'come', 'take', 'use', 'would', 'can', 'one', 'mr', 
    'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 
    'something', 'Watch', 'CNET', 'Video', 'Fox', 'Update', 'guardian', 'times',
    'business_insider', 'coronavirus', 'cnet', 'says', 'speaks', 'watch_live',
    'abc_news', 'york_times']
#COVID_KEYWORDS = [
#    'coronavirus', 'covid', 'covid-19', 'covid19', 'virus', 'masks', 'lockdown', 'quarantine'

#]
CORES = 4
num_topics = 10


from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def clean(df):
    # clean df
    if 'stories_id' in df:
        df = df.drop_duplicates(subset=['stories_id'])
    df = df.dropna(subset=['title'])
    df = df.sample(frac=args.sample)
    sentences = df['title'].replace(r'\\n',' ', regex=True)
    df['title'] = [strip_tags(s) for s in df['title']]
    return df


def get_dict_corpus(sentences):
    # Tokenize and remove punctuation
    sentences = [gensim.utils.simple_preprocess(s, deacc=True) for s in sentences]
    print("done preprocessing")
    #all_words = list(itertools.chain.from_iterable(sentences))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(sentences, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[sentences], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    words_bigrams = make_bigrams(sentences)
    words_trigrams = make_trigrams(sentences)

    swords = stopwords.words('English') + OTHER_STOPWORDS
    words_trigrams_no_stop = [[w for w in sent if w not in swords and len(w) > 1] for sent in words_trigrams]

    dictionary_LDA = corpora.Dictionary(words_trigrams_no_stop)
    corpus = [dictionary_LDA.doc2bow(w) for w in words_trigrams_no_stop]
    return dictionary_LDA, corpus


def sentences_to_gdsmm(sentences):
    dictionary, corpus = get_dict_corpus(sentences)
    # corpus is the list of documents in token, BOW format.
    # a sequence of tuples where the first entry is the token ID, and the second is the count of that token
    corpus_tokens = [[a[0] for a in sent] for sent in corpus]
    max_token = max([max(a) for a in corpus_tokens])
    mgp = MovieGroupProcess(K=8, alpha=0.1, beta=0.1, n_iters=30)
    mgp.fit(corpus_tokens, max_token)
    topics = mgp.cluster_word_distribution
    # array of BOW dicts of token ids
    scores = [mgp.score(sentence) for sentence in corpus]
    # return topics represented as dicts of word => val
    # return scores represented as array (len docs) of arrays (len topics)
    mapping = {v: k for k, v in dictionary.token2id.items()}
    topics_words = [
        {
            mapping[key]: val for key, val in dist.items()
        } for dist in topics
    ]
    return topics_words, scores



def sentences_to_topic_model(sentences):
    # first, clean:
    dictionary, corpus = get_dict_corpus(sentences)
    model = models.LdaModel(
        corpus, 
        num_topics=num_topics, 
        id2word=dictionary, 
        chunksize=8000,
        passes=8, 
        alpha='auto',
        eta='auto',
    )
    scores = [model[c] for c in corpus]
    topic_ndarray = model.get_topics()
    # convert the ndarray to dict
    mapping = {v: k for k, v in dictionary.token2id.items()}
    topics = []
    for row in topic_ndarray:
        topic = {}
        for i, val in enumerate(row):
            if val > 0:
                topic[mapping[i]] = val
    return topics, scores

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Train topic model on input text'
    )
    parser.add_argument('-train', dest='train', required=True,
                        help='filename for training')
    parser.add_argument('-sample', dest='sample', required=False, type=float,
            default=1, help='sample rate')
    parser.add_argument('-num-topics', dest='num_topics', required=False, type=int,
        default=20, help='number of topics')
    parser.add_argument('-method', dest='method', required=False, 
        choices=['LDA', 'GDSMM'], default='GDSMM',
        help='topic model to use')
    args = parser.parse_args()

    logging.info(f"Opening training file {args.train}")
    model = None
    dictionary = None
    corpus = None
    sentences = None
    df = pd.read_csv(args.train, sep='\t') 

    df = clean(df)
    sentences = df['title']


    logging.info(f"Training model...")
    if (args.method == 'LDA'):
        topics, scores = sentences_to_topic_model(sentences)
    else:
        topics, scores = sentences_to_gdsmm(sentences)

    # export the topic scores to headline_topics.tsv
    df = pd.DataFrame(scores).fillna(0)
    df['dominant_topic'] = np.argmax(df.values, axis=1)
    df['title'] = sentences.reset_index()['title']
    df.to_csv('headline_topics.tsv', sep='\t', index=False)

    # export the topic descriptions to topics.json
    with open('topics.json', 'wt') as f:
        f.write(json.dumps(topics))

