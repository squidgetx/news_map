# Python script to take an input corpus and output topics
import inflect
import pickle
import util
import tracery
from tracery.modifiers import base_english

import scipy
import json
import itertools
import fileinput
import logging
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from textblob import TextBlob

from gensim import corpora

from collections import defaultdict
import matplotlib.pyplot as plt

# import topic2tsne
from names import getFile, Datafile
import names

import pdb


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def rank(arr, val):
    # given value 0-1 get nearest index from array
    val = clamp(val, 0, 0.99)
    return arr[int(val * len(arr))]


PENINSULA = [
    "peninsula",
    "outcrop",
    "cape",
    "beach",
    "promontory",
    "peninsula",
]

LAND_NAMES = [
    "knoll",
    "hill",
    "bluff",
    "bayou",
    "desert",
    "flat",
    "field",
    "meadow",
    "plain",
    "grassland",
    "dune",
    "glen",
    "marsh",
    "swamp",
    "badland",
    "tundra",
    "plateau",
    "gorge",
    "glacier",
    "expanse",
    "wasteland",
    "canyon",
    "mountain",
    "peak",
]

ISLANDS = ["isle", "island", "island", "island", "mini-Continent"]


def score_sentiment(sentence):
    t = TextBlob(sentence)
    polarity = t.sentiment.polarity
    subjectivity = t.sentiment.subjectivity
    return polarity


def get_degrees(name, topics_json):
    # Maybe this code should live in cluster_topics?

    with open(getFile(name, Datafile.GRAPH_PICKLE), "rb") as f:
        graph = pickle.load(f)
        for topic in topics_json:
            try:
                topics_json[topic]["degree"] = graph.degree[topic]
            except:
                pdb.set_trace()
        return topics_json


def get_name(topics_json, topic):
    p = inflect.engine()
    if topics_json[topic]["size"] == 0:
        return ""
    # Sort by the second value in the tuple which is the float representation
    # of the weight
    top_cw = sorted(
        topics_json[topic]["common_words"], key=lambda k: float(k[1]), reverse=True
    )[0][0].split("_")
    top_rw = sorted(
        topics_json[topic]["relevant_words"], key=lambda k: float(k[1]), reverse=True
    )[0][0].split("_")
    # Some simple heuristics:
    # Put the shorter one first
    # Only use one if it's a triple

    # Multiplex by node degree, elevation, and alliteration
    if topics_json[topic]["degree"] == 0:
        land_names = ISLANDS
    elif topics_json[topic]["degree"] == 1:
        land_names = PENINSULA
    else:
        land_names = LAND_NAMES
    land_names = [w.capitalize() for w in land_names]
    size = util.scale(topics_json[topic]["size"], 0, 4000, 0, 1, use_clamp=True)

    words = ["#w#/#r#", "#r#/#w#", "#r#-#w#", "#w#-#r#", "#w#", "#w#", "#w#", "#w#"]
    if top_cw == top_rw:
        words = ["#w#"]
    elif set(top_cw).issubset(set(top_rw)) or len(top_rw) >= 3:
        words = ["#r#"]
    elif set(top_rw).issubset(set(top_cw)) or len(top_cw) >= 3 or size > 0.75:
        words = ["#w#"]
    rules = {
        "origin": ["#land# of #words#", "#words# #land#"],
        "words": words,
        "r": " ".join([w.capitalize() for w in top_rw]),
        "w": " ".join([w.capitalize() for w in top_cw]),
        "land": ["#l#", "#ls#"],
        "l": util.rank(land_names, size),
        "ls": p.plural(util.rank(land_names, size)),
    }
    grammar = tracery.Grammar(rules)
    grammar.add_modifiers(base_english)
    return grammar.flatten("#origin#")


def get_word_relevance(name, topics_json):
    print("Calcuating word relevance")

    dictionary = corpora.Dictionary.load(getFile(name, Datafile.DICTIONARY))
    LAMBDA = 0.9
    topic_ndarray = np.load(getFile(name, Datafile.TOPIC_NDARRAY))
    ps_token_corpus = np.array(
        [dictionary.cfs[token] / dictionary.num_pos for token in dictionary.keys()]
    )
    for topic, row in enumerate(topic_ndarray):
        if topic not in topics_json:
            continue
        sum_topic = np.sum(row)
        topic_word_relevance = (
            row / sum_topic * LAMBDA + (1 - LAMBDA) * row / sum_topic / ps_token_corpus
        )
        top_relevant_tokens = np.argsort(topic_word_relevance)[::-1][0:20]
        top_common_tokens = np.argsort(row)[::-1][0:20]
        BLACKLIST = [
            "ma_zone_forecast",
            "lottery_state_by",
            "mobile_world",
            "ct_boston_norton",
            "richard_grenell",
            "east_africa",
            "credit_cards",
        ]
        topics_json[topic]["relevant_words"] = [
            [dictionary.id2token[tok], topic_word_relevance[tok]]
            for tok in top_relevant_tokens
            if dictionary.id2token[tok] not in BLACKLIST
        ]
        topics_json[topic]["common_words"] = [
            [dictionary.id2token[tok], topic_word_relevance[tok]]
            for tok in top_common_tokens
            if dictionary.id2token[tok] not in BLACKLIST
        ]
        # del topics_json[str(topic)]["words"]

    return topics_json


def analyze_topics(name, headlines=None, scores=None):
    if headlines == None:
        headlines = pd.read_csv(open(getFile(name, Datafile.HEADLINES_TSV)), sep="\t")
    if scores == None:
        scores = pd.read_csv(open(getFile(name, Datafile.SCORES)), sep="\t")

    # rps = np.genfromtxt(getFile(name, Datafile.RUST_PROBABILITIES), delimiter=",")
    # headlines = headlines[headlines["title"].notnull()].reset_index()
    assert len(headlines) == len(scores)

    scores.columns = scores.columns.astype(int)
    scores_sums = scores.sum()
    scores_sums = scores_sums.sort_index()

    sentiments = np.zeros(len(headlines))

    """
    for i, row in headlines.iterrows():
        try:
            sentiments[i] = score_sentiment(row["title"])
        except:
            pdb.set_trace()
    """

    headlines["subjectivity"] = sentiments

    subj_map = headlines.groupby("dominant_topic").mean("subjectivity")
    count_map = headlines.groupby("dominant_topic").count()["title"]
    # get normalized count by media_name
    media_diversity = (
        headlines.groupby(["dominant_topic", "media_name"])
        .count()["title"]
        .unstack()
        .fillna(0)
        .apply(lambda x: x / np.sum(x))
        .apply(lambda x: scipy.stats.mstats.gmean(x) / np.mean(x), axis=1)
    )
    assert (scores_sums.index == subj_map.index).all()
    subj_map["media_diversity"] = media_diversity
    subj_map["count"] = count_map
    subj_map["size"] = scores_sums
    subj_map.to_csv(getFile(name, Datafile.TOPIC_METADATA_TSV), sep="\t")
    records = subj_map.to_dict(orient="index")
    for topic in subj_map.index:
        recent_headlines = scores[topic][scores[topic] > 0.9999].iloc[::-1][0:100]
        try:
            hdf = headlines.iloc[recent_headlines.index][
                ["title", "url", "media_name", "publish_date"]
            ]
            records[topic]["articles"] = hdf.to_dict(orient="records")
        except:
            pdb.set_trace()

    records = get_degrees(name, records)
    records = get_word_relevance(name, records)

    for topic in records:
        records[topic]["region_name"] = get_name(records, topic)

    topic_json = json.load(open(getFile(name, Datafile.TOPIC_JSON)))
    for topic in topic_json:
        topic_json[topic].update(records[int(topic)])

    with open(getFile(name, Datafile.TOPIC_JSON), "wt") as f:
        f.write(json.dumps(topic_json))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LDA topic model on input text")
    parser.add_argument(
        "-name", dest="name", required=True, help="filename for training"
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
    args = parser.parse_args()
    name = names.getName(args.name, args.start, args.interval)
    analyze_topics(name)
