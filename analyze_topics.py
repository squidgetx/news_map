# Python script to take an input corpus and output topics
import json
import itertools
import fileinput
import logging
import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import topic2tsne

import pdb

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Train LDA topic model on input text'
    )
    parser.add_argument('-filename', dest='filename', required=True,
                        help='filename for training')
    args = parser.parse_args()

    COLS = [str(i) for i in range(10)]
    df = pd.read_csv(args.filename, sep='\t')[COLS]
    dominant_topics = df.idxmax(axis=1)
    # TODO more sophisticated? 

    # Build a graph where each vertex is a topic, and there is an edge between topics 
    # if there are headline that share both topics
    # The edges are weighted by the strength of connection between them
    # Each vertex is also weighted, so the (vertex_weight) of topic 0 
    # is the sum of the vertex_weights of all its neighbors, plus all topics that 

    topics = list(df.columns)

    adj = pd.DataFrame()
    # just build the adjacency matrix?
    for t in topics:
        adj[t] = np.sum(df[dominant_topics == t])
    
    plt.imshow(adj, cmap='hot', interpolation='nearest')
    #plt.show()
    adj.to_csv('analysis.tsv', sep='\t')

    def find_dominant_topic(thresh):
        def finder(r):
            values = np.sort(r)[::-1]
            # max is first element, second element is second, etc.
            if values[0] > values[1] * thresh:
                return np.argmax(r)
            return np.NAN
        return finder

    dts = df.apply(find_dominant_topic(2), axis=1)
    print('With 2', dts.isna().sum() / len(dts))
    dts = df.apply(find_dominant_topic(1.5), axis=1)
    print('With 1.5', dts.isna().sum() / len(dts))
    dts = df.apply(find_dominant_topic(1.2), axis=1)
    print('With 1.2', dts.isna().sum() / len(dts))
    dts = df.apply(find_dominant_topic(1.01), axis=1)
    print('With 1.01', dts.isna().sum() / len(dts))

    # write json 
    nodes = [{"id": t, "group": 1} for t in topics]
    links = []
    for t, row in adj.iterrows():
        for t2 in range(len(row)):
            if (t == str(t2)):
                print ('skip')
                continue # don't write the diagonals
            links.append({
                "source": t,
                "target": str(t2),
                "value": row[t2]/row[int(t)],
            })
    with open('topic_adjacency.json', 'wt') as f:
        json.dump({'nodes': nodes, 'links': links}, f)
            


