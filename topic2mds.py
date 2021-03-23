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

CORES = 4
NAMED_COLS = ["title", "dominant_topic"]
USE_FITSNE = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LDA topic model on input text")
    parser.add_argument(
        "-distance-file",
        dest="distance_file",
        help="file for topic distances",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.distance_file, sep="\t", index_col=0)
    embedding = manifold.MDS(dissimilarity="precomputed")
    transformed = embedding.fit_transform(df)
    pd.DataFrame(transformed).to_csv(f"{args.distance_file[0:-4]}.mds.tsv", sep="\t")
    stress = np.sqrt(
        embedding.stress_ / ((embedding.dissimilarity_matrix_.ravel() ** 2).sum() / 2)
    )
    # https://stackoverflow.com/questions/36428205/stress-attribute-sklearn-manifold-mds-python
    print(f"Stress: {stress}")
    print("[Poor > 0.2 > Fair > 0.1 > Good > 0.05 > Excellent > 0.025 > Perfect > 0.0]")
