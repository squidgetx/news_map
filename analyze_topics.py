# Python script to take an input corpus and output topics
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


def get_edges(distances_ndarray, sizes):
    # which edges to keep:
    # start assigning edges sorted by closeness
    # cost of an edge assignment is sum of node degrees * weight
    # connecting 2 islands: low cost
    # connecting 2 continents: higher cost
    # if the edge assignment cost is too high, skip it

    # maybe we actually need a 3-layer joining system to reduce the number of unrelated islands
    # cuz realistically we can have as many islands as we want off to the side or whatever
    distances = []
    for i, row in enumerate(distances_ndarray):
        for j, value in enumerate(row):
            if j <= i:
                continue
            distances.append((i, j, value))
    distances = sorted(distances, key=lambda x: x[2])

    edges = []
    degrees = np.zeros(len(distances_ndarray))
    for dist in distances:
        if dist[2] >= 0.9:
            continue
        if sizes[dist[0]] == 1:
            continue
        cost = (degrees[dist[0]] + degrees[dist[1]]) * dist[2]

        # sum_degrees = 2 * 0.8 => 1.6
        # sum_degrees = 3 * 0.8 => 2.4
        threshold = 3
        if cost < threshold:
            degrees[dist[0]] += 1
            degrees[dist[1]] += 1
            edges.append((dist[0], dist[1], dist[2] * 2))

    return edges


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
            if len(edges[pair["a"]]) < 4:
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
        if pair["distance"] > 0.7 and pair["distance"] < 0.75:
            if len(edges[pair["a"]]) < 1 and len(edges[pair["b"]]) < 1:
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


def build_and_save_graph(name, prevname=None):

    df = pd.read_csv(getFile(name, Datafile.DISTANCE_JS), sep="\t", index_col=0)
    df.columns = df.columns.astype(int)

    # create array of distance pairs
    topics = list(df.columns)
    edges = []
    for row in df.iterrows():
        for topic in topics:
            edges.append({"a": int(row[0]), "b": int(topic), "distance": row[1][topic]})
    data = build_graph(edges)
    with open(getFile(name, Datafile.TOPIC_ADJACENCY), "wt") as f:
        json.dump(data, f)

    if prevname:
        with open(getFile(prevname, Datafile.LAYOUT), "rt") as f:
            prevData = json.loads(f)["layouts"]
        currentDict = corpora.Dictionary.load(getFile(name, Datafile.DICTIONARY))
        prevDict = corpora.Dictionary.load(getFile(prevname, Datafile.DICTIONARY))
    else:
        layout_graph(data, df, name)


def layout_graph_with_predecessor(data, distances, name, predDistances, predPositions):
    # given a set of metaclusters
    # and a previous set of metaclusters, with their positions
    # layout their positions

    # 1. calculate distances between the old and the new metaclusters
    # 2. Some metaclusters will be close to the older metaclusters
    # 3. How should we measure distances? Weighted mean of JS divergence?
    # 4. For close ones, set the position to the old position
    # 5. For far ones, do what? idk
    # maybe, each cluster is given initial position where old clusters are
    # then, run small clustering algorithm for X number of steps
    # then, run large clustering algortihm for Y number of steps

    # predPositions is a dict of positions for each cluster
    initial = {}
    for i in predPositions:
        # figure out which topic is closest
        j = getClosest(i, predDistances)
        initial[j] = predPositions[i]

    return None


def layout_graph(data, distances, name, initPositions=None, microIter=50, macroIter=70):
    print("laying out graph")
    graph = nx.Graph()
    graph.add_nodes_from([n for n in range(len(distances))])
    graph.add_weighted_edges_from(
        [(n["source"], n["target"], n["value"]) for n in data["links"]]
    )
    components = sorted(nx.connected_components(graph), key=len, reverse=True)

    # Each component is like a continent.
    # this function attempts to lay out continents using a naive greedy algorithm
    # we assume that there aren't that many continents that should be "near" each other.
    layouts = []
    centers = []
    radii = []
    compgraphs = []

    for i, comp in enumerate(components):
        compgraph = graph.subgraph(comp).copy()
        compgraphs.append(compgraph)
        if initPositions:
            positions = {
                k: initPositions[k] for k in initPositions if k in compgraph.nodes
            }
            cx = np.mean([positions[k][0] for k in positions])
            cy = np.mean([positions[k][1] for k in positions])
            centers.append([cx, cy])
            for k in positions:
                positions[k][0] -= cx
                positions[k][1] -= cy
        else:
            centers.append(None)
        layout = nx.spring_layout(
            compgraph, scale=len(comp) ** 0.5, pos=initPositions, iterations=microIter
        )
        layout_list = [
            {
                "id": key,
                "x": layout[key][0],
                "y": layout[key][1],
            }
            for key in layout
        ]
        layouts.append(layout_list)

        radius = 0
        for pos in layout_list:
            radius = max(radius, (pos["x"] ** 2 + pos["x"] ** 2) ** 0.5)
        radii.append(radius)

    print("laying out second layer")
    # Second layer force layout:
    cg_distances = np.zeros((len(compgraphs), len(compgraphs)))
    for i, cg1 in enumerate(compgraphs):
        for j, cg2 in enumerate(compgraphs):
            if j <= i:
                continue
            node_distances = []
            for node1 in cg1:
                for node2 in cg2:
                    node_distances.append(distances[node1][node2])
            cg_distances[i][j] = np.mean(node_distances)

    sizes = [len(cg) for cg in compgraphs]
    cg_edges = get_edges(cg_distances, sizes)

    compgraph_nx = nx.Graph()
    # compgraph_nx.add_nodes_from([i for i in range(len(compgraphs))])
    compgraph_nx.add_weighted_edges_from(cg_edges)
    if initPositions:
        initMetaPositions = {i: c for i, c in enumerate(centers)}
    else:
        initMetaPositions = None
    cg_layout = nx.spring_layout(
        compgraph_nx, scale=30, iterations=macroIter, pos=initMetaPositions
    )
    for l in cg_layout:
        centers[l] = cg_layout[l].tolist()

    for i, layout in enumerate(layouts):
        for pos in layout:
            if centers[i]:
                pos["x"] += centers[i][0] + 30
                pos["y"] += centers[i][1] + 30

    with open(getFile(name, Datafile.LAYOUT), "wt") as f:
        f.write(json.dumps({"layouts": layouts, "centers": centers}))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train LDA topic model on input text")
    parser.add_argument(
        "-name", dest="name", required=True, help="filename for training"
    )
    args = parser.parse_args()
    name = args.name
    build_and_save_graph(name)
