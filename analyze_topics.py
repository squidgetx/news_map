# Python script to take an input corpus and output topics
import random
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
import topic2tsne
from names import getFile, Datafile

import pdb

land_names = [
    "archipelago",
    "continent" "fjord",
    "island",
    "islands",
    "peninsula",
    "badlands",
    "bayou",
    "beach",
    "bluff",
    "canyon",
    "cape",
    "cirque",
    "col",
    "desert",
    "flat",
    "fields",
    "meadows",
    "plains",
    "dunes",
    "glacier",
    "glen",
    "gorges",
    "hills",
    "knoll",
    "marsh",
    "swamp",
    "plateau",
    "grassland",
    "tundra",
]


def score_sentiment(sentence):
    t = TextBlob(sentence)
    polarity = t.sentiment.polarity
    subjectivity = t.sentiment.subjectivity
    return subjectivity


def analyze_topics(name, headlines=None, scores=None):
    if headlines == None:
        headlines = pd.read_csv(open(getFile(name, Datafile.HEADLINES_TSV)), sep="\t")
    if scores == None:
        scores = pd.read_csv(open(getFile(name, Datafile.SCORES)), sep="\t")

    headlines = headlines[headlines["title"].notnull()].reset_index()

    scores_sums = scores.sum()
    scores_sums.index = scores_sums.index.astype(int)
    scores_sums = scores_sums.sort_index()

    sentiments = np.zeros(len(headlines))

    for i, row in headlines.iterrows():
        try:
            sentiments[i] = score_sentiment(row["title"])
        except:
            pdb.set_trace()

    headlines["subjectivity"] = sentiments

    subj_map = headlines.groupby("dominant_topic").mean("subjectivity")
    count_map = headlines.groupby("dominant_topic").count()["title"]
    # get normalized count by media_name
    media_diversity = (
        headlines.groupby(["dominant_topic", "media_name"])
        .count()["index"]
        .unstack()
        .fillna(0)
        .apply(lambda x: x / np.sum(x))
        .apply(lambda x: scipy.stats.mstats.gmean(x) / np.mean(x), axis=1)
    )
    subj_map["media_diversity"] = media_diversity
    subj_map["count"] = count_map
    subj_map["size"] = scores_sums
    subj_map.to_csv(getFile(name, Datafile.TOPIC_METADATA_TSV), sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LDA topic model on input text")
    parser.add_argument(
        "-name", dest="name", required=True, help="filename for training"
    )
    args = parser.parse_args()
    name = args.name
    analyze_topics(name)
