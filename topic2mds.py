# Python script to take an input corpus and output topics
from nltk.tokenize import sent_tokenize
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
from names import Datafile, getFile

CORES = 4
NAMED_COLS = ["title", "dominant_topic"]
USE_FITSNE = False


def calculateMDS(name):
    df = pd.read_csv(getFile(name, Datafile.DISTANCE_JS), sep="\t", index_col=0)
    embedding = manifold.MDS(dissimilarity="precomputed")
    transformed = embedding.fit_transform(df)
    pd.DataFrame(transformed).to_csv(getFile(name, Datafile.MDS), sep="\t")
    pd.DataFrame(transformed).to_csv(f"data/distancesJS.mds.tsv", sep="\t")
    stress = np.sqrt(
        embedding.stress_ / ((embedding.dissimilarity_matrix_.ravel() ** 2).sum() / 2)
    )
    # https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python
    print(f"Stress: {stress}")
    print("[Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]")
    for row in df.iterrows():
        for i, value in enumerate(row[1]):
            if i < row[0]:
                continue
            if value > 0.7 and value < 0.75:
                print(row[0], i, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LDA topic model on input text")
    parser.add_argument(
        "-name",
        dest="name",
        help="file for topic distances",
    )
    args = parser.parse_args()
    calculateMDS(args.name)
