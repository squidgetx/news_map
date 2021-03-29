# Python script to take an input corpus and output topics
import json
import itertools
import fileinput
import logging
import argparse
import pandas as pd
import numpy as np

from collections import defaultdict
import matplotlib.pyplot as plt
import topic2tsne
from names import getFile, Datafile

import pdb


def build_graph(distances):
    """
    given distance array, build graph that we will convert into continent locations
    assume distances array of dicts?
    """
    edges = defaultdict(list)
    # phase 1:
    for pair in distances:
        if pair["distance"] < 0.6 and pair["distance"] > 0:
            existing_edges = [e["target"] for e in edges[pair["a"]]]
            if pair["b"] in existing_edges:
                continue
            edges[pair["a"]].append({"target": pair["b"], "value": pair["distance"]})
            edges[pair["b"]].append({"target": pair["a"], "value": pair["distance"]})
    for pair in distances:
        if pair["distance"] < 0.65 and pair["distance"] > 0.6:
            if len(edges[pair["a"]]) < 3:
                existing_edges = [e["target"] for e in edges[pair["a"]]]
                if pair["b"] in existing_edges:
                    continue
                edges[pair["a"]].append(
                    {"target": pair["b"], "value": pair["distance"]}
                )
                edges[pair["b"]].append(
                    {"target": pair["a"], "value": pair["distance"]}
                )

    for pair in distances:
        if pair["distance"] > 0.65 and pair["distance"] < 0.7:
            if len(edges[pair["a"]]) < 2:
                existing_edges = [e["target"] for e in edges[pair["a"]]]
                if pair["b"] in existing_edges:
                    continue
                edges[pair["a"]].append(
                    {"target": pair["b"], "value": pair["distance"]}
                )
                edges[pair["b"]].append(
                    {"target": pair["a"], "value": pair["distance"]}
                )

    for pair in distances:
        if pair["distance"] > 0.7 and pair["distance"] < 0.75:
            if len(edges[pair["a"]]) < 1:
                existing_edges = [e["target"] for e in edges[pair["a"]]]
                if pair["b"] in existing_edges:
                    continue
                edges[pair["a"]].append(
                    {"target": pair["b"], "value": pair["distance"]}
                )
                edges[pair["b"]].append(
                    {"target": pair["a"], "value": pair["distance"]}
                )

    links = []
    for node in edges:
        links.extend(
            [
                {"source": node, "target": n["target"], "value": n["value"]}
                for n in edges[node]
            ]
        )
    return {"nodes": [{"id": node_id} for node_id in edges], "links": links}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train LDA topic model on input text")
    parser.add_argument(
        "-name", dest="name", required=True, help="filename for training"
    )
    args = parser.parse_args()
    name = args.name

    df = pd.read_csv(getFile(name, Datafile.DISTANCE_JS), sep="\t", index_col=0)

    # create array of distance pairs
    topics = list(df.columns)
    edges = []
    for row in df.iterrows():
        for topic in topics:
            edges.append({"a": int(row[0]), "b": int(topic), "distance": row[1][topic]})
    data = build_graph(edges)
    with open(getFile(name, Datafile.TOPIC_ADJACENCY), "wt") as f:
        json.dump(data, f)

    """
    dominant_topics = df.idxmax(axis=1)
    # TODO more sophisticated?

    # Build a graph where each vertex is a topic, and there is an edge between topics
    # if there are headline that share both topics
    # The edges are weighted by the strength of connection between them
    # Each vertex is also weighted, so the (vertex_weight) of topic 0
    # is the sum of the vertex_weights of all its neighbors, plus all topics that


    # just build the adjacency matrix?
    for t in topics:
        adj[t] = np.sum(df[dominant_topics == t])

    plt.imshow(adj, cmap="hot", interpolation="nearest")
    # plt.show()
    adj.to_csv("analysis.tsv", sep="\t")

    def find_dominant_topic(thresh):
        def finder(r):
            values = np.sort(r)[::-1]
            # max is first element, second element is second, etc.
            if values[0] > values[1] * thresh:
                return np.argmax(r)
            return np.NAN

        return finder

    dts = df.apply(find_dominant_topic(2), axis=1)
    print("With 2", dts.isna().sum() / len(dts))
    dts = df.apply(find_dominant_topic(1.5), axis=1)
    print("With 1.5", dts.isna().sum() / len(dts))
    dts = df.apply(find_dominant_topic(1.2), axis=1)
    print("With 1.2", dts.isna().sum() / len(dts))
    dts = df.apply(find_dominant_topic(1.01), axis=1)
    print("With 1.01", dts.isna().sum() / len(dts))

    # write json
    nodes = [{"id": t, "group": 1} for t in topics]
    links = []
    for t, row in adj.iterrows():
        for t2 in range(len(row)):
            if t == str(t2):
                print("skip")
                continue  # don't write the diagonals
            links.append(
                {
                    "source": t,
                    "target": str(t2),
                    "value": row[t2] / row[int(t)],
                }
            )
    with open("topic_adjacency.json", "wt") as f:
        json.dump({"nodes": nodes, "links": links}, f)
    """
