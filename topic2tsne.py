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
from sklearn import manifold , decomposition
from fitsne import FItSNE 
import pdb

CORES = 4
NAMED_COLS = ['title', 'dominant_topic']
USE_FITSNE = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train LDA topic model on input text'
    )
    parser.add_argument('-method', dest='method', required=True, default='TSNE',
        choices=['TSNE', 'FITSNE', 'PCA'],
        help='2D method for visualization')
    args = parser.parse_args()

    arr = pd.read_csv('headline_topics.tsv', sep='\t')
    # tSNE Dimension Reduction
    vals = arr.drop(NAMED_COLS, axis=1).values.copy(order='C')
    if args.method == 'FITSNE':
        tsne_lda = FItSNE(
            vals,
            perplexity=20000 * 0.04,
            theta=0.5,
        )
    elif args.method == 'TSNE': 
        model = manifold.TSNE(
            n_components=2, 
            verbose=1, 
            random_state=0, 
            init='pca',
            perplexity=20000 * 0.04,
            angle=0.9,
            n_iter=1500,
            n_jobs = CORES,
        )
        tsne_lda = model.fit_transform(vals)
    else:
        model = decomposition.PCA(n_components=2)
        tsne_lda = model.fit_transform(vals)
        pdb.set_trace()

    df = pd.DataFrame(tsne_lda)
    df['title'] = arr['title']
    df['dominant_topic'] = arr['dominant_topic']
    df['x'] = df[0]
    df['y'] = df[1]
    df.to_csv(f'{args.method}.tsv', sep='\t')
    print(f'{args.method}.tsv written')

