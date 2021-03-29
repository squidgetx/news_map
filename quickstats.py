# Python script to take an input corpus and output topics
import scipy
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import itertools
from gensim import corpora
from gensim import models
from gensim.test.utils import datapath
import fileinput
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn import manifold, decomposition
from fitsne import FItSNE
import pdb
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LDA topic model on input text")
    parser.add_argument(
        "-name",
        dest="name",
        help="name of stories file",
    )
    parser.add_argument(
        "-thresh",
        dest="thresh",
        type=float,
        default=0.6,
        required=False,
        help="threshold for topic distances",
    )
    args = parser.parse_args()
    name = args.name
    df = pd.read_csv(f"distancesJS_{name}.tsv", sep="\t", index_col=0)
    for row in df.iterrows():
        for i, value in enumerate(row[1]):
            if i < row[0]:
                continue
            if value > 0 and value < args.thresh:
                print(row[0], i, value)
    with open(f"topics_{name}.json", "rt") as f:
        topics = json.load(f)
        print("\n")
    totals = pd.DataFrame([row["total"] for row in topics], columns=["value"])
    print(scipy.stats.describe(totals["value"]))
    plt.hist(totals, bins=100)
    # plt.show()
    df_sorted = totals.sort_values(ascending=False, by="value")
    np.set_printoptions(suppress=True)
    print(df_sorted[df_sorted["value"] > 100].reset_index().values)
