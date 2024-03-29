# Python script to take an input corpus and output topics
import argparse
import json
import logging
import sys
import time
import datetime
from pathlib import Path

import gensim
import gensim.downloader as api
from gensim import corpora
from gensim import models
from gensim.test.utils import datapath

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import os

from gsdmm import MovieGroupProcess
from scipy.spatial.distance import jensenshannon

# import topic2mds
import cluster_topics
from analyze_topics import analyze_topics
import subprocess
from names import Datafile, getFile
import names
from wmdistance import WMDistance

import pdb

from io import StringIO
from html.parser import HTMLParser

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
    "new",
    "update",
    "could",
    "HKO_WHL",
]

CORES = 4


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
    df.dropna(subset=["title"], inplace=True)
    df["title"] = df["title"].replace(r"\\n", " ", regex=True)
    df["title"] = [strip_tags(s) for s in df["title"]]
    df["title"] = [None if s.startswith("http") else s for s in df["title"]]
    df = df[df["title"].notnull()]

    return df


def get_dict_corpus(sentences):
    # Tokenize and remove punctuation
    print(f"Preprocessing {len(sentences)} rows...")
    sentences = [gensim.utils.simple_preprocess(s, deacc=True) for s in sentences]
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

    # words_bigrams = make_bigrams(sentences)
    words_trigrams = make_trigrams(sentences)

    swords = stopwords.words("English") + stopwords.words("Spanish") + OTHER_STOPWORDS
    words_trigrams_no_stop = [
        [w for w in sent if w not in swords and len(w) > 1] for sent in words_trigrams
    ]

    dictionary_LDA = corpora.Dictionary(words_trigrams_no_stop)
    dictionary_LDA.filter_extremes(no_below=5, no_above=0.5)

    corpus = [dictionary_LDA.doc2bow(w) for w in words_trigrams_no_stop]
    print("done preprocessing")
    return dictionary_LDA, corpus


def sentences_to_gsdmm_rust(dictionary, corpus, num_topics, name, alpha=0.1, beta=0.1):
    # prepare files for consumption by rust executable:
    # vocabfile: one token per line
    print("preparing input for gsdmm-rust")
    with open("data/vocabfile.txt", "wt") as f:
        for t in dictionary.itervalues():
            f.write(t)
            f.write("\n")
    index = []
    with open("data/sentences.txt", "wt") as f:
        for i, doc in enumerate(corpus):
            arr = []
            for tok in doc:
                for _ in range(tok[1]):
                    arr.append(dictionary.id2token[tok[0]])
            if arr:
                f.write(" ".join(arr))
                f.write("\n")
                index.append(i)

    print("spawning gsdmm-rust subprocess")
    # spawn the rust subprocess
    stream_p = subprocess.Popen(
        [
            "gsdmm-rust/target/release/gsdmm",
            "data/sentences.txt",
            "data/vocabfile.txt",
            f"data/{name}.",
            "-k",
            str(num_topics),
            "-a",
            f"{alpha}",
            "-b",
            f"{beta}",
            "-m",
            "50",
        ],
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    while True:
        output = stream_p.stdout.readline()
        if stream_p.poll() is not None:
            break
        if output:
            print(output.strip())

    # now read the cluster descriptions file into an ndarray
    print("retrieving gsdmm-rust output")
    mapping = dictionary.token2id
    topic_ndarray = np.zeros((num_topics, len(mapping)))

    with open(getFile(name, Datafile.RUST_CLUSTER_DESC), "rt") as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            line = line.split(" ")
            cluster_i = int(line[0])
            cluster_words = line[1:]
            for pair in cluster_words:
                comps = pair.split(":")
                token = comps[0]
                val = int(comps[1])
                tokid = mapping[token]
                topic_ndarray[cluster_i][tokid] = val

    scores = []
    with open(getFile(name, Datafile.RUST_LABELS), "rt") as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            comps = line.split(",")
            scores.append({int(comps[0]): float(comps[1])})
    scores = pd.DataFrame(scores).fillna(0)
    scores.index = index
    return dictionary, topic_ndarray, scores


def sentences_to_gsdmm(sentences, num_topics):
    # return dictionary, topic ndarray, and scores for each document
    dictionary, corpus = get_dict_corpus(sentences)
    # corpus is the list of documents in token, BOW format.
    # a sequence of tuples where the first entry is the token ID,
    # and the second is the count of that token

    corpus_tokens = [[a[0] for a in sent] for sent in corpus]
    max_token = max([max(a) for a in corpus_tokens if a])
    mgp = MovieGroupProcess(K=num_topics, alpha=0.1, beta=0.5, n_iters=50)
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
    """
    given a topic distrubtion represented as a dictionary and ndarray,
    return a dict of token => val mapping
    """
    mapping = {v: k for k, v in dictionary.token2id.items()}
    topics = []
    for row in topic_ndarray:
        topic = {}
        for i, val in enumerate(row):
            if val > 0:
                topic[mapping[i]] = int(val)
        topics.append(topic)
    return topics


def get_topic_distances_JS(topics, topics2):
    """
    input: ndarray for each topic
    output: n_topic * n_topic ndarray of jensen shannon distances for each topic
    n_topic is the larger of the two dimensions, so the returned array is symmetric
        with zeros in places where there is no valid topic comparison
    """
    threshold = 50
    dim = max(len(topics), len(topics2))

    distances = np.zeros((dim, dim))
    for i in range(len(topics)):
        thresh_i = np.sort(topics[i])[-threshold]
        topic_i = topics[i].copy()
        topic_i[topic_i < thresh_i] = 0
        for j in range(len(topics2)):
            thresh_j = np.sort(topics2[j])[-threshold]
            topic_j = topics2[j].copy()
            topic_j[topic_j < thresh_j] = 0
            distances[i][j] = jensenshannon(topic_i, topic_j)

    return np.nan_to_num(distances, nan=1.0)


def get_topic_distances_WMD(model, dictionary, topics):
    """
    input: w2v model, ndarray for each topic, dictionary for id2token mapping
    # TODO: even if we use WMD we need a fallback case for when
    # one of the words isn't included in the W2V

    # Create a whitelist of tokens,
    # Keep tokens that are "significant" in any topic
    # Throw away tokens that are "insignificant" in every topic
    # tokens are considered significant if they compose at least THRESHOLD
    # of the total topic distribution.

    # We do this because WMD is O(N^2), where N = total vocabulary size,
    # so reducing the vocabulary size dramatically improves performance

    # Trimming insignificant words shouldn't affect precision too significantly
    # given that we already know that topics should have a relatively
    # asymmetric token distribution
    """

    threshold = 100

    tokens_to_keep = set()
    for row in topics:
        for tokid, num in enumerate(row):
            tk = row.argsort()[-threshold:]
            for t in tk:
                tokens_to_keep.add(t)
            # if num / np.sum(row) > threshold:
            #    tokens_to_keep.add(tokid)

    wmd = WMDistance(model, dictionary, tokens_to_keep)

    start = time.time()
    distances = np.zeros((len(topics), len(topics)))
    for i in range(len(topics)):
        for j in range(i, len(topics)):
            distances[i][j] = wmd.get(topics[i], topics[j])
            distances[j][i] = distances[i][j]
    print(f"{time.time() - start}")

    """
    topic_dict = topic_ndarray_to_dict(dictionary, topics)
    topic_sentences = []
    for topic in topic_dict:
        sentence = []
        for word, freq in topic.items():
            for _ in range(int(freq / 4)):
                sentence.append(word)
        topic_sentences.append(sentence)

    distances = np.zeros((len(topics), len(topics)))
    # N^2 / 2 now for topics, but N-topics should be low (<1K)
    for i, sen1 in enumerate(topic_sentences):
        for j in range(i + 1, len(topic_sentences)):
            sen2 = topic_sentences[j]
            distances[i][j] = wmdistance(model, sen1, sen2)
            distances[j][i] = distances[i][j]
    """

    return distances


def get_topics(
    df, dictionary, corpus, num_topics, name, method="GSDMM-Rust", alpha=0.1, beta=0.1
):
    assert len(df) == len(corpus)
    if method == "LDA":
        dictionary, topics, scores = sentences_to_topic_model(
            sentences, args.num_topics
        )
    elif method == "GSDMM-Rust":
        dictionary, topics, scores = sentences_to_gsdmm_rust(
            dictionary, corpus, args.num_topics, name, alpha=alpha, beta=beta
        )
    elif method == "GSDMM":
        dictionary, topics, scores = sentences_to_gsdmm(sentences, args.num_topics)

    print("exporting dictionary and nparray")

    # export everything
    dictionary.save(getFile(name, Datafile.DICTIONARY))
    np.save(getFile(name, Datafile.TOPIC_NDARRAY), topics)

    print("preparing scores")
    scores_df = scores
    assert (scores.index == df.index).all()
    scores_df.to_csv(getFile(name, Datafile.SCORES), sep="\t", index=False)
    scores_sums = scores_df.sum()
    scores_sums = scores_sums.sort_index()
    # Write this here because the graph algo step requires this info
    # and the analyze_topics.py as it is right now depends on the graph algo
    # output
    records = pd.DataFrame(scores_sums, columns=["size"]).to_dict(orient="index")
    with open(getFile(name, Datafile.TOPIC_JSON), "wt") as f:
        f.write(json.dumps(records))

    scores_df["dominant_topic"] = scores_df.idxmax(axis=1)
    scores_df["title"] = df["title"]
    assert scores_df["title"].notnull().all()
    scores_df["url"] = df["url"]
    scores_df["publish_date"] = df["publish_date"]
    if "media_name" not in df:
        df["media_name"] = df["domain"]
    media_names = df["media_name"].fillna("No Media Name")
    scores_df["media_name"] = media_names
    scores_df[["dominant_topic", "title", "media_name", "url", "publish_date"]].to_csv(
        getFile(name, Datafile.HEADLINES_TSV), sep="\t", index=False
    )
    assert len(scores_df) == len(corpus)

    return topics


def calculate_intertopic_distances(dictionary, topics, dictionary2, topics2):
    """
    calculate the jensen shannon distances between two given
    topic distrubtions with different dictionaries

    needs some careful munging, because the dictionaries that
    the topic distribution is defined over is not the same.

    returns: ndarray of intertopic distances
    """

    # create a merged dictionary and use this to re-map the topic2 ndarray
    transformer = dictionary.merge_with(dictionary2)

    transformed_topics_2 = np.zeros((len(topics2), len(dictionary)))
    for i, row in enumerate(topics2):
        doc = [(col, val) for col, val in enumerate(row)]
        transformed_doc = transformer[doc]
        for item in transformed_doc:
            transformed_topics_2[i][item[0]] = item[1]

    # copy the first ndarray over, leaving zeros in empty spaces
    transformed_topics_1 = np.zeros((len(topics), len(dictionary)))
    for i, row in enumerate(topics):
        for j, val in enumerate(row):
            transformed_topics_1[i][j] = val

    ret = get_topic_distances_JS(transformed_topics_1, transformed_topics_2)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train topic model on input text")
    parser.add_argument("-name", dest="name", help="filename for training", default=names.Basename.US_MAINSTREAM)
    parser.add_argument("-load", dest="load", help="name for loading")
    parser.add_argument(
        "-sample",
        dest="sample",
        required=False,
        type=float,
        default=1,
        help="sample rate",
    )
    parser.add_argument(
        "-start",
        dest="start",
        help="start date (ISO)",
    )
    parser.add_argument(
        "-interval",
        dest="interval",
        default=28,
        type=int,
        help="Number of days to include after the given start date",
    )
    parser.add_argument(
        "-step",
        type=int,
        dest="step",
        required=False,
        help="Number of days forward to compare.",
    )
    parser.add_argument(
        "-num-topics",
        dest="num_topics",
        required=False,
        type=int,
        default=100,
        help="number of topics",
    )
    parser.add_argument(
        "-alpha",
        dest="alpha",
        required=False,
        type=float,
        default=0.1,
        help="gsdmm alpha",
    )
    parser.add_argument(
        "-beta",
        dest="beta",
        required=False,
        type=float,
        default=0.1,
        help="gsdmm beta",
    )
    parser.add_argument(
        "-method",
        dest="method",
        required=False,
        choices=["LDA", "GSDMM", "GSDMM-Rust"],
        default="GSDMM-Rust",
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
    topics = []
    name = ""
    if args.name:
        basename = names.Basename[args.name]
        name = names.getName(
            basename,
            datetime.datetime.fromisoformat(args.start), 
            args.interval
        )
        Path(os.path.dirname("data/" + name)).mkdir(parents=True, exist_ok=True)

        logging.info(f"Opening data files...")
        dnames = names.getDataNames(basename, args.start, args.interval)
        df = pd.concat((pd.read_csv(f, sep="\t", memory_map=True) for f in dnames))
        print(f"Reading {len(df)} rows...")
        df = clean(df)

        print(f"Training model...")
        dictionary, corpus = get_dict_corpus(df["title"])

        # Trim corpus to remove empty documents
        nonempty = [i for i, val in enumerate(corpus) if val != []]
        corpus = [i for i in corpus if i != []]
        df = df.iloc[nonempty].reset_index()

        topics = get_topics(
            df,
            dictionary,
            corpus,
            args.num_topics,
            name,
            alpha=args.alpha,
            beta=args.beta,
        )

        print(f"saving topics and dictionaries...")
        dictionary.save(getFile(name, Datafile.DICTIONARY))
        np.save(getFile(name, Datafile.TOPIC_NDARRAY), topics)

    elif args.load:
        basename = names.Basename[args.load]
        name = names.getName(basename, args.start, args.interval)
        topics = np.load(getFile(name, Datafile.TOPIC_NDARRAY))
        dictionary = corpora.Dictionary.load(getFile(name, Datafile.DICTIONARY))
    else:
        print("You need to either use the load or name arguments")
        sys.exit()

    name2 = None
    if args.step:
        print("calculating intertopic distances")
        name2 = names.getPrevName(basename, args.start, args.interval, args.step)
        topic2_filename = getFile(name2, Datafile.TOPIC_NDARRAY)
        topic2_dict = getFile(name2, Datafile.DICTIONARY)
        if os.path.exists(topic2_filename) and os.path.exists(topic2_dict):
            topics2 = np.load(topic2_filename, allow_pickle=True)
            dictionary2 = corpora.Dictionary.load(topic2_dict)
            interdistances = calculate_intertopic_distances(
                dictionary, topics, dictionary2, topics2
            )
            pd.DataFrame(interdistances).to_csv(
                getFile(name, Datafile.INTERDISTANCE_JS), sep="\t"
            )
        else:
            print(
                f"Could not calculate intertopic distances! {topic2_filename} or {topic2_dict} not found"
            )

    print("calculating JS distances")
    distancesJS = get_topic_distances_JS(topics, topics)
    pd.DataFrame(distancesJS).to_csv(getFile(name, Datafile.DISTANCE_JS), sep="\t")

    if args.wmd:
        print("Loading W2V model...")
        model = api.load("word2vec-google-news-300")
        distancesWMD = get_topic_distances_WMD(model, dictionary, topics)
        pd.DataFrame(distancesWMD).to_csv(
            getFile(name, Datafile.DISTANCE_WMD), sep="\t"
        )

    print("building topic adjacency graph")
    if name2:
        cluster_topics.build_and_save_graph(name, prevname=name2)
    else:
        cluster_topics.build_and_save_graph(name)

    analyze_topics(name)
