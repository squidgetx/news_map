# Python script to take an input corpus and output topics
import subprocess
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
import names
import util
from names import getFile, Datafile

import pdb


"""
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

    return graph
"""


def build_graph(df, sizes: dict):
    """
    Given distance DF and list of valid topic indexes,
    build initial graph with weighted edges
    Return nx.graph object
    """
    graph = nx.Graph()

    for i in sizes:
        graph.add_node(i, size=sizes[i])

    topics = list(sizes.keys())

    for i, topic_a in enumerate(topics):
        for j, topic_b in enumerate(topics):
            if j <= i:
                continue
            distance = df[topic_a][topic_b]
            assert distance > 0
            if distance < 0.75:
                graph.add_edge(topic_a, topic_b, weight=distance)

    return graph


def build_and_save_graph(name, prevname=None):

    df = pd.read_csv(getFile(name, Datafile.DISTANCE_JS), sep="\t", index_col=0)
    df.columns = df.columns.astype(int)

    # get the sizes
    sizes = names.getTopicSizes(name)
    topics = list(sizes.keys())

    graph = build_graph(df, sizes)

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
                        "center": center if center else {"x": 0, "y": 0},
                        "topic": topic["id"],
                    }
                    centers[topic["id"]] = center
        interDistances = pd.read_csv(
            open(getFile(name, Datafile.INTERDISTANCE_JS)), sep="\t", index_col=0
        )
        layout_graph_with_predecessor(
            graph, df, name, interDistances, prevLayout, centers
        )
        util.run_subprocess(
            [
                "node",
                "layout.js",
                "-n",
                f"{name}",
                "-i",
                "4",
            ],
        )

    else:
        layout_graph(graph, df, name)
        util.run_subprocess(
            ["node", "layout.js", "-n", f"{name}", "-i", "1000", "--grid"],
        )


def layout_graph_with_predecessor(
    graph, distances, name, interDistances, prevLayout, prevCenters
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
            initial[topic] = {"position": [0, 0], "center": {"x": 0, "y": 0}}
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
    layout_graph(graph, distances, name, initial, prevLayout, microIter=16, macroIter=4)


def export_components(components, name="metaclusters.json"):
    export = []
    for idx, nxgraph in enumerate(components):
        nodes = []
        for t in nxgraph.nodes.data():
            node = {"id": t[0]}
            node.update(t[1])
            nodes.append(node)
        links = []
        for i, t in enumerate(nxgraph.edges.data()):
            link = {"source": t[0], "target": t[1]}
            link.update(t[2])
            links.append(link)
        export.append({"nodes": nodes, "edges": links, "id": idx})

    with open(name, "wt") as f:
        f.write(json.dumps(export))


def export_graph(graph, name="metametaclusters.json"):
    nodes = []
    for i, t in enumerate(graph.nodes.data()):
        node = {"id": t[0]}
        node.update(t[1])
        nodes.append(node)
    links = []
    for i, t in enumerate(graph.edges.data()):
        link = {"source": t[0], "target": t[1]}
        link.update(t[2])
        links.append(link)
    export = {"nodes": nodes, "edges": links}
    with open(name, "wt") as f:
        f.write(json.dumps(export))


def decompose(graph, flow_threshold=4.5, ratio_threshold=0.05):
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
                and sizes[0] > 1
                and sizes[1] > 1
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
    for it in range(n):
        no_cuts = True
        new_components = []
        for graph in components:
            cut = decompose(graph)
            if cut is None:
                new_components.append(graph)
                continue
            no_cuts = False
            print(it, cut)
            subgraph1 = graph.subgraph(cut[1][0]).copy()
            subgraph2 = graph.subgraph(cut[1][1]).copy()
            cut_edges = graph.edges() - subgraph1.edges() - subgraph2.edges()
            for edge in list(cut_edges):
                edges.append(
                    {
                        "source": edge[0],
                        "target": edge[1],
                        "weight": 8 / (it ** 0.5 + 1),
                    }
                )
            new_components.append(subgraph1)
            new_components.append(subgraph2)
        components = new_components
        if no_cuts:
            print("No cuts made in last iteration, stopping early!")
            return components, edges
    return components, edges


def layout_graph(
    graph,
    distances,
    name,
    initPositions=None,
    prevLayout=None,
    microIter=50,
    macroIter=70,
):
    nx.write_gpickle(graph, getFile(name, Datafile.GRAPH_PICKLE))

    print("forming metaclusters")
    if initPositions:
        for i in initPositions:
            node = initPositions[i]
            graph.nodes[i]["x"] = node["position"][0]  # - node["center"]["x"]
            graph.nodes[i]["y"] = node["position"][1]  # - node["center"]["y"]
            graph.nodes[i]["initial_group"] = node.get("group")
            graph.nodes[i]["initial_topic"] = node.get("topic")

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    compgraphs = [graph.subgraph(comp).copy() for comp in components]
    # min cuts
    compgraphs, edges = cut_components(compgraphs, n=10)

    print("forming metametaclusters")
    # Second layer force layout:
    # Use the edges that were trimmed from the min-cut process
    # first make a lookup table for node_id => component graph index
    node_lookup = {}
    for i, graph in enumerate(compgraphs):
        for node in list(graph.nodes):
            node_lookup[node] = i
            graph.nodes[node]["group"] = i
            graph.nodes[node]["degree"] = graph.degree[node]
            graph.nodes[node]["type"] = "core"
            if graph.nodes[node]["degree"] == 1:
                graph.nodes[node]["type"] = "leaf"
            if graph.nodes[node]["degree"] == 0:
                graph.nodes[node]["type"] = "unconnected"
            for edge in edges:
                if edge["source"] == node:
                    graph.nodes[node]["type"] = "bridge"
                if edge["target"] == node:
                    graph.nodes[node]["type"] = "bridge"

    # then, we can just iterate through the cut edges
    mmc_graph = nx.Graph()
    mmc_graph.add_nodes_from([i for i in range(len(compgraphs))])
    for edge in edges:
        # always go small -> large
        c1 = min(node_lookup[edge["source"]], node_lookup[edge["target"]])
        c2 = max(node_lookup[edge["source"]], node_lookup[edge["target"]])
        if c2 in mmc_graph[c1]:
            mmc_graph[c1][c2]["weight"] /= 2
        else:
            mmc_graph.add_edge(c1, c2, weight=10)

    # mmc_graph = get_edges(mmc_graph, compgraphs, distances)
    if initPositions:
        for i, graph in enumerate(compgraphs):
            cluster_centers = [
                [initPositions[node]["center"]["x"], initPositions[node]["center"]["y"]]
                for node in list(graph.nodes)
            ]
            try:
                center = np.mean(cluster_centers, axis=0)
            except:
                pdb.set_trace()
            mmc_graph.nodes[i]["x"] = center[0]
            mmc_graph.nodes[i]["y"] = center[1]

    export_components(compgraphs, getFile(name, Datafile.METACLUSTERS))
    print("components exported")
    export_graph(mmc_graph, getFile(name, Datafile.METAMETACLUSTERS))
    print("metametaclusters exported")
    with open(getFile(name, Datafile.FULL_GRAPH), "wt") as f:
        f.write(json.dumps(edges))
    return

    """
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
    """


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
