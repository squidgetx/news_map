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
from scipy.spatial.distance import jensenshannon

import pdb

OTHER_STOPWORDS = [
    "say",
    "get",
    "go",
    "know",
    "may",
    "need",
    "like",
    "make",
    "see",
    "want",
    "come",
    "take",
    "use",
    "would",
    "can",
    "one",
    "mr",
    "bbc",
    "image",
    "getty",
    "de",
    "en",
    "caption",
    "also",
    "copyright",
    "something",
    "Watch",
    "CNET",
    "Video",
    "Fox",
    "Update",
    "guardian",
    "times",
    "business_insider",
    "coronavirus",
    "cnet",
    "says",
    "speaks",
    "watch_live",
    "fox_business",
    "abc_news",
    "york_times",
    "daily_mail_online",
    "boston_globe",
]

CORES = 4

from io import StringIO
from html.parser import HTMLParser


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
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
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def clean(df):
    # clean df
    if "stories_id" in df:
        df = df.drop_duplicates(subset=["stories_id"])
    df = df.dropna(subset=["title"])
    df = df.sample(frac=args.sample)
    sentences = df["title"].replace(r"\\n", " ", regex=True)
    df["title"] = [strip_tags(s) for s in df["title"]]
    return df


def get_dict_corpus(sentences):
    # Tokenize and remove punctuation
    sentences = [gensim.utils.simple_preprocess(s, deacc=True) for s in sentences]
    print("done preprocessing")
    # all_words = list(itertools.chain.from_iterable(sentences))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(
        sentences, min_count=5, threshold=100
    )  # higher threshold fewer phrases.
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

    swords = stopwords.words("English") + stopwords.words("Spanish") + OTHER_STOPWORDS
    words_trigrams_no_stop = [
        [w for w in sent if w not in swords and len(w) > 1] for sent in words_trigrams
    ]

    dictionary_LDA = corpora.Dictionary(words_trigrams_no_stop)
    print(len(dictionary_LDA))
    dictionary_LDA.filter_extremes()
    print(len(dictionary_LDA))

    corpus = [dictionary_LDA.doc2bow(w) for w in words_trigrams_no_stop]
    return dictionary_LDA, corpus


def sentences_to_gdsmm(sentences, num_topics):
    # return dictionary, topic ndarray, and scores for each document
    dictionary, corpus = get_dict_corpus(sentences)
    # corpus is the list of documents in token, BOW format.
    # a sequence of tuples where the first entry is the token ID,
    # and the second is the count of that token

    corpus_tokens = [[a[0] for a in sent] for sent in corpus]
    max_token = max([max(a) for a in corpus_tokens if a])
    mgp = MovieGroupProcess(K=num_topics, alpha=0.1, beta=0.1, n_iters=30)
    mgp.fit(corpus_tokens, max_token)
    topics = mgp.cluster_word_distribution
    # array of BOW dicts of token ids
    scores = [mgp.score(sentence) for sentence in corpus]
    # return topics represented as dicts of word => val
    # return scores represented as array (len docs) of arrays (len topics)
    mapping = {v: k for k, v in dictionary.token2id.items()}

    # create ndarray from topic map
    topic_ndarray = np.zeros((len(topics), max_token + 1))
    for i, topic in enumerate(topics):
        for k, v in topic.items():
            topic_ndarray[i][k] = v

    return dictionary, topic_ndarray, scores


def sentences_to_topic_model(sentences, num_topics):
    # first, clean:
    dictionary, corpus = get_dict_corpus(sentences)
    model = models.LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        chunksize=8000,
        passes=8,
        alpha="auto",
        eta="auto",
    )
    scores = [model[c] for c in corpus]
    topic_ndarray = model.get_topics()
    # convert the ndarray to dict

    return dictionary, topic_ndarray, scores


def topic_ndarray_to_dict(dictionary, topic_ndarray):
    mapping = {v: k for k, v in dictionary.token2id.items()}
    topics = []
    for row in topic_ndarray:
        topic = {}
        for i, val in enumerate(row):
            if val > 0:
                topic[mapping[i]] = int(val)
        topics.append(topic)
    return topics


def get_topic_distances_JS(topics):
    # input: ndarray for each topic
    # output: n_topic * n_topic ndarray of jensen shannon distances for each topic

    distances = np.zeros((len(topics), len(topics)))
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            distances[i][j] = jensenshannon(topics[i], topics[j])
            distances[j][i] = distances[i][j]

    return distances


def get_topic_distances_WMD(model, dictionary, topics):
    # input: w2v model, ndarray for each topic, dictionary for id2token mapping
    # TODO: even if we use WMD we need a fallback case for when
    # one of the words isn't included in the W2V

    # first convert all sentencecs to array of tokens
    topic_dict = topic_ndarray_to_dict(dictionary, topics)
    topic_sentences = []
    for topic in topic_dict:
        sentence = []
        for word, freq in topic.items():
            for _ in range(int(freq / 4)):
                sentence.append(word)
        topic_sentences.append(sentence)
    import pdb

    pdb.set_trace()

    distances = np.zeros((len(topics), len(topics)))
    # N^2 / 2 now for topics, but N-topics should be low (<1K)
    for i, sen1 in enumerate(topic_sentences):
        for j in range(i + 1, len(topic_sentences)):
            sen2 = topic_sentences[j]
            distances[i][j] = model.wmdistance(sen1, sen2)
            distances[j][i] = distances[i][j]

    return distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train topic model on input text")
    parser.add_argument(
        "-train", dest="train", required=True, help="filename for training"
    )
    parser.add_argument(
        "-sample",
        dest="sample",
        required=False,
        type=float,
        default=1,
        help="sample rate",
    )
    parser.add_argument(
        "-num-topics",
        dest="num_topics",
        required=False,
        type=int,
        default=20,
        help="number of topics",
    )
    parser.add_argument(
        "-method",
        dest="method",
        required=False,
        choices=["LDA", "GDSMM"],
        default="GDSMM",
        help="topic model to use",
    )
    parser.add_argument(
        "-wmd",
        dest="wmd",
        required=False,
        action="store_const",
        const=True,
        help="whether or not to use WMD formatting",
    )
    args = parser.parse_args()

    logging.info(f"Opening training file {args.train}")

    df = pd.read_csv(args.train, sep="\t")
    df = clean(df)
    sentences = df["title"]

    logging.info(f"Training model...")
    if args.method == "LDA":
        dictionary, topics, scores = sentences_to_topic_model(
            sentences, args.num_topics
        )
    else:
        dictionary, topics, scores = sentences_to_gdsmm(sentences, args.num_topics)

    print(topics.shape)

    # export the topic scores to headline_topics.tsv
    df = pd.DataFrame(scores).fillna(0)
    df["dominant_topic"] = np.argmax(df.values, axis=1)
    df["title"] = sentences.reset_index()["title"]
    df.to_csv("headline_topics.tsv", sep="\t", index=False)
    totals = df.sum()

    # export the topic descriptions to topics.json
    topics_dict = topic_ndarray_to_dict(dictionary, topics)
    for k in range(len(topics_dict)):
        topics_dict[k]["total"] = totals[int(k)]
    with open("topics.json", "wt") as f:
        f.write(json.dumps(topics_dict))

    distancesJS = get_topic_distances_JS(topics)
    pd.DataFrame(distancesJS).to_csv("distancesJS.tsv", sep="\t")

    if args.wmd:
        print("Loading W2V model...")
        model = api.load("word2vec-google-news-300")
        distancesWMD = get_topic_distances_WMD(model, dictionary, topics)
        pd.DataFrame(distancesWMD).to_csv("distancesWMD.tsv", sep="\t")
