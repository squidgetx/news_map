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


def get_distances(graphs, distances):
    cg_distances = np.zeros((len(graphs), len(graphs)))
    for i, cg1 in enumerate(graphs):
        for j, cg2 in enumerate(graphs):
            if j <= i:
                continue
            node_distances = []
            for node1 in cg1:
                for node2 in cg2:
                    node_distances.append(distances[node1][node2])
            cg_distances[i][j] = np.mean(node_distances)
            cg_distances[j][i] = np.mean(node_distances)
    return cg_distances


def get_edges(graph, components, topic_distances):
    # components: array of metacluster graph
    # graph: graph of nodes, node i is component[i]
    # iteratively add edges between jh
    distances_ndarray = get_distances(components, topic_distances)
    distances = []
    for i, row in enumerate(distances_ndarray):
        for j, value in enumerate(row):
            if j <= i:
                continue
            distances.append((i, j, value))

    distances = sorted(distances, key=lambda x: x[2])
    unconnected_nodes = []
    while True:
        metacomponents = sorted(nx.connected_components(graph), key=len, reverse=True)
        if len(metacomponents) == 1:
            break
        # This is the smallest connected component
        component_to_place = metacomponents[-1]
        # Iterate through its nodes to find the best connection to another node
        # Best connection: smallest distance edge that doesnt already exist
        score = 1.0
        candidate_idx = None
        for node in component_to_place:
            # Where should this go?
            # Attach it to the best node
            candidates = distances_ndarray[node]
            for i, val in enumerate(candidates):
                if val == 0:
                    continue
                if i in graph[node]:
                    continue
                if val < score:
                    score = val
                    candidate_idx = (node, i)
        if candidate_idx:
            print(f"Placing edge {candidate_idx}")
            graph.add_edge(candidate_idx[0], candidate_idx[1], weight=val)
        else:
            # Did not find any candidates. This means this node literally
            # has JS distance 1 to every other cluster
            for node in component_to_place:
                assert sum(distances_ndarray[node]) == len(distances_ndarray) - 1
                graph.remove_node(node)
                unconnected_nodes.append(node)

    for node in unconnected_nodes:
        selected = random.choice(list(graph.nodes()))
        graph.add_edge(selected, node, weight=3)
    return graph


def score_topic(topic):
    # given a topic distrubution, return a 2D shape describing
    # its XY position on the dimensions human/world and insignificant/significant
    return


def score_metacluster(topics, sizes, topic_ndarray):
    # given a metacluster, score it on our arbitrary x/y plane
    #
    return


def build_graph(distances, n_nodes):
    """
    given distance array, build graph that we will convert into continent locations
    assume distances array of dicts?
    """
    edges = defaultdict(list)
    # phase 1:
    for pair in distances:
        if pair["distance"] < 0.75 and pair["distance"] > 0:
            existing_edges = [e["target"] for e in edges[pair["a"]]]
            if pair["b"] in existing_edges:
                continue
            edges[pair["a"]].append({"target": pair["b"], "value": pair["distance"]})
            edges[pair["b"]].append({"target": pair["a"], "value": pair["distance"]})
    for pair in distances:
        if pair["distance"] > 0.75 and pair["distance"] < 0.8:
            if len(edges[pair["a"]]) < 1 or len(edges[pair["b"]]) < 1:
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
    return {
        "nodes": [{"id": node_id} for node_id in range(n_nodes)],
        "links": links,
    }


def build_and_save_graph(name, prevname=None):

    df = pd.read_csv(getFile(name, Datafile.DISTANCE_JS), sep="\t", index_col=0)
    df.columns = df.columns.astype(int)

    # create array of distance pairs
    topics = list(df.columns)
    edges = []
    for row in df.iterrows():
        for topic in topics:
            edges.append({"a": int(row[0]), "b": int(topic), "distance": row[1][topic]})
    data = build_graph(edges, len(df))
    with open(getFile(name, Datafile.TOPIC_ADJACENCY), "wt") as f:
        json.dump(data, f)

    # get the sizes
    topic_json = json.load(open(getFile(name, Datafile.TOPIC_JSON)))
    sizes = {}
    for i, topic in enumerate(topic_json):
        sizes[i] = topic["_metadata_"]["total"]
    for node in data["nodes"]:
        node["size"] = sizes[node["id"]]

    if prevname:
        prevLayout = {}
        centers = {}
        with open(getFile(prevname, Datafile.LAYOUT), "rt") as f:
            prevData = json.load(f)
            for i, group in enumerate(prevData["layouts"]):
                for topic in group:
                    center = prevData["centers"][i]
                    prevLayout[topic["id"]] = {
                        "position": [topic["x"], topic["y"]],
                        "group": i,
                        "center": center if center else [0, 0],
                        "topic": topic["id"],
                    }
                    centers[topic["id"]] = center
        interDistances = pd.read_csv(
            open(getFile(name, Datafile.INTERDISTANCE_JS)), sep="\t", index_col=0
        )
        layout_graph_with_predecessor(
            data, df, name, interDistances, prevLayout, centers
        )
    else:
        layout_graph(data, df, name)


def layout_graph_with_predecessor(
    data, distances, name, interDistances, prevLayout, prevCenters
):
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
    # then, run large clustering algorithm for Y number of steps

    def get_topic(distances, i: int):
        row = interDistances.iloc[i]
        minDistance = 1.0
        minTopic = None
        for index, ele in enumerate(row):
            if index == i:
                continue
            if ele < minDistance:
                minDistance = ele
                minTopic = index
        return minTopic, minDistance

    # prevLayout is a dict of positions for each cluster
    initial = {}
    for topic in range(len(distances)):
        # figure out which topic is closest
        prevTopic, distance = get_topic(interDistances, topic)
        if prevTopic == None:
            print(f"{topic} distance was None !!!")
            initial[topic] = {"position": [0, 0], "center": [0, 0]}
        else:
            initial[topic] = prevLayout[prevTopic]
            initial[topic]["distance"] = distance
    # Sanity Check
    # Current problem is that initial[62] shows that
    # topic number 62 should connect to ptopic number 50,
    # but it seems obvious that we should actually connect
    # to ptopic number 60 instead?
    # Can't figure out what's wrong with it though???
    # It seems like the interdistance code is corret, so maybe
    # we need to double check the intertopic JS distance code
    # :(
    # ...
    topics = json.load(open(getFile(name, Datafile.TOPIC_JSON)))
    prevname = "us_mainstream_stories_trunc_2020-02-01_2020-02-29"
    ptopics = json.load(open(getFile(prevname, Datafile.TOPIC_JSON)))
    layout_graph(data, distances, name, initial, prevLayout, microIter=16, macroIter=4)


def show_top(topics, n):
    return sorted(topics[n].items(), key=lambda k: k[1] if k[0] != "_metadata_" else 0)[
        -10:
    ]


def export_components(components, name="metaclusters.json"):
    export = []
    for idx, nxgraph in enumerate(components):
        nodes = [t[1] for t in nxgraph.nodes.data()]
        links = []
        for i, t in enumerate(nxgraph.edges.data()):
            link = {"source": t[0], "target": t[1]}
            link.update(t[2])
            links.append(link)
        export.append({"nodes": nodes, "edges": links, "id": idx})

    with open(name, "wt") as f:
        f.write(json.dumps(export))


def export_graph(graph, name="metametaclusters.json"):
    nodes = [{"id": t[0]} for t in graph.nodes.data()]
    links = []
    for i, t in enumerate(graph.edges.data()):
        link = {"source": t[0], "target": t[1]}
        link.update(t[2])
        links.append(link)
    export = {"nodes": nodes, "edges": links}
    with open(name, "wt") as f:
        f.write(json.dumps(export))


def decompose(graph, flow_threshold=3, ratio_threshold=0.15):
    """
    given a graph with jensen shannon distances as the weights,
    return a min cut (as a 2-tuple of node IDs) if one exists
    that is less than the flow_threshold and the ratio of the two node sets are
    greater than the ratio_threshold.
    """
    # first, prepare the capacity attribute
    # we want it to be cheaper to remove edges with higher Jensen Shannon distance,
    # which is currently encoded as weight
    if len(graph) < 4:
        return None
    min_cuts = []
    all_cuts = []
    min_cut_data = []
    for edge in list(graph.edges):
        graph[edge[0]][edge[1]]["capacity"] = 1 / graph[edge[0]][edge[1]]["weight"]
    # pick a random target node
    for idx, target in enumerate(list(graph.nodes)):
        for source in list(graph.nodes)[idx:]:
            if target == source:
                continue
            cut = nx.algorithms.minimum_cut(graph, source, target)
            sizes = [len(cut[1][0]), len(cut[1][1])]
            sizes.sort()
            ratio = sizes[0] / sizes[1]
            if (
                cut[0] < flow_threshold
                and ratio > ratio_threshold
                and cut not in min_cuts
                and sizes[0] > 3
                and sizes[1] > 3
            ):
                min_cut_data.append([cut[0], ratio, len(min_cuts)])
                min_cuts.append(cut)
            if cut not in all_cuts:
                all_cuts.append([cut[0], ratio])
    if not min_cuts:
        return None
    min_cut = sorted(min_cut_data, key=lambda x: x[1])[-1]
    return min_cuts[min_cut[2]]


def cut_components(components, n=3):
    edges = []
    for _ in range(n):
        no_cuts = True
        new_components = []
        for graph in components:
            cut = decompose(graph)
            if cut is None:
                new_components.append(graph)
                continue
            no_cuts = False
            print(_, cut)
            subgraph1 = graph.subgraph(cut[1][0]).copy()
            subgraph2 = graph.subgraph(cut[1][1]).copy()
            cut_edges = graph.edges() - subgraph1.edges() - subgraph2.edges()
            edges.extend(list(cut_edges))
            new_components.append(subgraph1)
            new_components.append(subgraph2)
        components = new_components
        if no_cuts:
            print("No cuts made in last iteration, stopping early!")
            return components, edges
    return components, edges


def layout_graph(
    data,
    distances,
    name,
    initPositions=None,
    initCenters=None,
    microIter=50,
    macroIter=70,
):
    print("forming metaclusters")
    graph = nx.Graph()
    graph.add_nodes_from([(node["id"], node) for node in data["nodes"]])
    graph.add_weighted_edges_from(
        [(n["source"], n["target"], n["value"]) for n in data["links"]]
    )
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    compgraphs = [graph.subgraph(comp).copy() for comp in components]
    compgraphs, edges = cut_components(compgraphs, n=10)

    # min cuts
    export_components(compgraphs)
    print("components exported")
    print("forming metametaclusters")
    # Second layer force layout:
    # Use the edges that were trimmed from the min-cut process
    # first make a lookup table for node_id => component graph index
    node_lookup = {}
    for i, graph in enumerate(compgraphs):
        for node in list(graph.nodes):
            node_lookup[node] = i
    # then, we can just iterate through the cut edges
    mmc_graph = nx.Graph()
    mmc_graph.add_nodes_from([i for i in range(len(compgraphs))])
    for edge in edges:
        # always go small -> large
        c1 = min(node_lookup[edge[0]], node_lookup[edge[1]])
        c2 = max(node_lookup[edge[0]], node_lookup[edge[1]])
        if c2 in mmc_graph[c1]:
            mmc_graph[c1][c2]["weight"] += 1
        else:
            mmc_graph.add_edge(c1, c2, weight=1)

    # legacy method
    mmc_graph = get_edges(mmc_graph, compgraphs, distances)
    for i, center in enumerate(initCenters):
        mmc_graph[i]["x"] = initCenters[i][0]
        mmc_graph[i]["y"] = initCenters[i][1]
    export_graph(mmc_graph)
    print("metametaclusters exported")
    return

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
            positions = {}
            cluster_centers = {}
            for k in compgraph.nodes:
                if k not in initPositions:
                    pdb.set_trace()
                print(k, initPositions[k])
                cluster_centers[k] = [
                    initPositions[k]["center"][0],
                    initPositions[k]["center"][1],
                ]
                positions[k] = [
                    initPositions[k]["position"][0],
                    initPositions[k]["position"][1],
                ]
            cx = np.mean([cluster_centers[k][0] for k in cluster_centers])
            cy = np.mean([cluster_centers[k][1] for k in cluster_centers])
            centers.append([cx, cy])

            # Rescale positions to roughly fit in 0, 1 interval.
            # We don't want to be that sensitive to outliers, so we'll center to the mean and
            # rescale to the 25th and 75th percentiles
            cx25 = np.quantile([positions[k][0] for k in positions], 0.25)
            cx50 = np.quantile([positions[k][0] for k in positions], 0.50)
            cx75 = np.quantile([positions[k][0] for k in positions], 0.75)
            cy25 = np.quantile([positions[k][1] for k in positions], 0.25)
            cy50 = np.quantile([positions[k][1] for k in positions], 0.50)
            cy75 = np.quantile([positions[k][1] for k in positions], 0.75)
            for k in positions:
                if cx75 > cx25:
                    positions[k][0] = (positions[k][0] - cx50) / (cx75 - cx25) * 0.5
                    positions[k][1] = (positions[k][1] - cy50) / (cy75 - cy25) * 0.5
                else:
                    positions[k][0] = positions[k][0] - cx50
                    positions[k][1] = positions[k][1] - cy50
        else:
            positions = None
            centers.append(None)

        if positions:
            layout = nx.spring_layout(
                compgraph, scale=len(comp) ** 0.8, pos=positions, iterations=microIter
            )
        else:
            layout = nx.spring_layout(compgraph, scale=len(comp) ** 0.8, iterations=50)
            # layout = layout_d3(compgraph)  # , scale=len(comp) ** 0.8, iterations=50)
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

    # With no scaling, then we should be able to cleanly start the simulation over again, roughly speaking.
    if initPositions:
        initMetaPositions = {i: c for i, c in enumerate(centers)}
        cg_layout = nx.spring_layout(
            compgraph_nx, scale=None, iterations=macroIter, pos=initMetaPositions
        )

    else:
        initMetaPositions = None
        cg_layout = nx.spring_layout(
            compgraph_nx, scale=None, iterations=macroIter, pos=initMetaPositions
        )
    for l in cg_layout:
        centers[l] = cg_layout[l].tolist()

    for i, layout in enumerate(layouts):
        for pos in layout:
            if centers[i]:
                pos["x"] += centers[i][0] * 30 + 30
                pos["y"] += centers[i][1] * 30 + 30

    with open(getFile(name, Datafile.LAYOUT), "wt") as f:
        f.write(json.dumps({"layouts": layouts, "centers": centers}))


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
