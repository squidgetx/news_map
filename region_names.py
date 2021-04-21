# Python script to take an input corpus and output topics
import random
import json
import itertools
import fileinput
import logging
import argparse
import pandas as pd
import numpy as np
import networkx as nx

from gensim import corpora

from collections import defaultdict
import matplotlib.pyplot as plt
import topic2tsne
from names import getFile, Datafile

import pdb
land_names = [
    'archipelago',
    'continent'
    'fjord',
    'island',
    'islands',
    'peninsula',
    'badlands',
    'bayou',
    'beach',
    'bluff',
    'canyon',
    'cape',
    'cirque',
    'col',
    'desert',
    'flat',
    'fields',
    'meadows',
    'plains',
    'dunes',
    'glacier',
    'glen',
    'gorges',
    'hills',
    'knoll',
    'marsh',
    'swamp',
    'plateau',
    'grassland',
    'tundra',
]

def get_name(topic):
    # dict of words, with frequency


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train LDA topic model on input text")
    parser.add_argument(
        "-name", dest="name", required=True, help="filename for training"
    )
    parser.add_argument(
        "-prevname", dest="prevname", required=False, help="filename for training"
    )
    args = parser.parse_args()
    name = args.name
    prevname = args.prevname
    build_and_save_graph(name, prevname=args.prevname)
