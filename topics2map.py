# Python script to take a tsv of coordinates as input and output map
# Heavily inspired by https://github.com/mewo2/deserts

import pdb
import networkx as nx
import scipy

import numpy as np
import itertools
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import scipy.stats as stats
import sys

from collections import defaultdict
from time import sleep

import names
from names import getFile, Datafile


def get_radius(n):
    # Needs to be synced with layout.js
    return (n["size"] ** 0.5) * 2 + 2


def plt_voronoi(pts, topics):
    if isinstance(pts, pd.DataFrame):
        pts = pts[["x", "y"]]
    fig = spatial.voronoi_plot_2d(
        spatial.Voronoi(pts),
        show_vertices=False,
        line_colors="orange",
        line_width=2,
        line_alpha=0.6,
        point_size=2,
    )
    ax = fig.gca()
    for i, topic in topics.iterrows():
        ax.add_patch(
            plt.Circle((topic["x"], topic["y"]), topic["radius"], color="#aaa")
        )
        # ax.text(topic["x"], topic["y"], int(topic["radius"] * largest), fontsize=6)

    fig.set_size_inches(5, 5)
    plt.xlim(-0.1, 1.1)
    plt.ylim(1.1, -0.1)
    plt.show()


class PolygonMap:
    # invariant: all points between 0-1
    adj_points = defaultdict(list)
    adj_vertices = defaultdict(list)

    headlines = []

    def get_rough_topic(self, idx):
        """Given an index of a rough point (given by rough voronoi
        object, or the rough KDTree, get the topic ID for the
        cell. Return -1 if no topic ID found (water)
        """
        return int(self.rough_points["topic"][idx])

    def create_initial_polygons(self, layout_df, density=2000, land_density=5000):
        """
        Given a set of initial topics, create the "rough" voronoi skeleton
        for the map.

        The density parameter controls roughly how fine grained this skeleton is.
        A higher parameter leads to more points
        """
        # layout is a dict of x, y positions for each topic
        self.topic_df = layout_df
        self.n_topics = len(self.topic_df)

        # First, calculate centers of each continent and the distances of each
        # region from continent center along with the max and median distance
        centers = layout_df.groupby("group").mean()[["x", "y"]]
        for g, center in centers.iterrows():
            layout_df.loc[layout_df["group"] == g, "gx"] = center["x"]
            layout_df.loc[layout_df["group"] == g, "gy"] = center["y"]
        layout_df["gdist"] = (
            (layout_df["x"] - layout_df["gx"]) ** 2
            + (layout_df["y"] - layout_df["gy"]) ** 2
        ) ** 0.5 + layout_df["radius"]

        max_dist = layout_df.groupby("group").max()["gdist"]
        medians = layout_df.groupby("group").median()["gdist"]

        # Second, spawn points basically in a ring around each continent.
        # Use a gaussian distribution so that there are the most points along the
        # outermost edge of the map, and fewer as we go inland. This
        # guarantees that the continent will be properly surrounded by water
        # while reducing the number of lakes that form inside (while still allowing) some

        noise_pts = np.ndarray((0, 2))
        for g, center in centers.iterrows():
            n_pts = int(max_dist.loc[g] ** 2 * np.pi * density)
            theta = np.random.random(n_pts) * np.pi * 2
            radius = np.random.normal(
                loc=max_dist.loc[g], scale=medians.loc[g] / 2, size=n_pts
            )
            x, y = (
                radius * np.cos(theta) + center["x"],
                radius * np.sin(theta) + center["y"],
            )
            noise_pts = np.concatenate((noise_pts, np.stack((x, y), axis=1)))
        plt_voronoi(noise_pts, layout_df)

        # Third, remove any noise points that directly lie inside of a region's
        # circular range.
        tree = spatial.KDTree(noise_pts)
        pts_to_remove = tree.query_ball_point(
            layout_df[["x", "y"]], layout_df["radius"]
        )
        remove_set = set()
        for p in pts_to_remove:
            remove_set.update(p)
        noise_pts = np.delete(noise_pts, list(remove_set), axis=0)
        noise_df = pd.DataFrame(noise_pts, columns=["x", "y"])
        noise_df["topic"] = -1

        # Fourth, generate extra points for each region/topic. This guarantees
        # that the region will have a land area closer to its given radius
        topic_points = []
        for i, row in layout_df.iterrows():
            n_pts = int(row["radius"] ** 2 * np.pi * land_density)
            topic_points.append([row["x"], row["y"], i])
            for _ in range(n_pts):
                theta = np.random.random() * 3.14 * 2
                rad = np.random.random() * row["radius"]
                x = row["x"] + np.cos(theta) * rad
                y = row["y"] + np.sin(theta) * rad
                topic_points.append([x, y, i])
        rough_topic_points = pd.DataFrame(topic_points, columns=["x", "y", "topic"])

        self.set_rough_points(rough_topic_points, noise_df)
        self.add_lakes()

    def create_lake_points(self, points):
        """
        Given a pair of X,Y coordinate tuples that represent a straight
        line segment, return a set of coordinates of a lake
        meant to replace that segment
        """
        N_LAKE_PTS = 10
        # Could get fancy with curved lakes here but we'll do the simple
        # thing for now.
        delta = (points[1] - points[0]) / N_LAKE_PTS
        return [points[0] + i * delta for i in range(N_LAKE_PTS)]

    def add_lakes(self):
        """
        Add lakes if required given the input graph
        """
        graph = nx.read_gpickle(getFile(name, Datafile.GRAPH_PICKLE))
        # Basically, we want to examine each Voronoi ridge
        # and if its corresponding edge doesn't exist in the input graph,
        # then we "kill" it with a lake
        extra_points = []
        for i, pair in enumerate(self.rough_voronoi.ridge_points):
            topicA = self.get_rough_topic(pair[0])
            topicB = self.get_rough_topic(pair[1])
            if topicA == -1 or topicB == -1:
                # One of them is a water cell, so w/e
                continue
            if topicA == topicB:
                # bordering yourself is OK
                continue
            if topicB in graph[topicA]:
                # The edge exists in the graph, so it's OK.
                continue
            # If we reach this point, we have a Voronoi ridge
            # that links two topics that SHOULD NOT have an edge
            ridge_vertices = self.rough_voronoi.vertices[
                self.rough_voronoi.ridge_vertices[i]
            ]
            extra_points.extend(self.create_lake_points(ridge_vertices))

        self.set_rough_points(
            self.rough_topic_points,
            pd.concat(
                (
                    self.rough_points,
                    pd.DataFrame(extra_points, columns=["x", "y"]),
                ),
                ignore_index=True,
            ),
        )

        plt_voronoi(self.rough_points, self.topic_df)

    def set_rough_points(self, topic_points, noise_pts):
        assert "x" in topic_points
        assert "y" in topic_points
        assert "topic" in topic_points
        assert "x" in noise_pts
        assert "y" in noise_pts
        assert "topic" in noise_pts
        noise_pts = noise_pts.fillna(-1)

        self.rough_points = pd.concat((topic_points, noise_pts), ignore_index=True)
        self.rough_voronoi = spatial.Voronoi(self.rough_points[["x", "y"]])
        self.rough_kdtree = spatial.KDTree(self.rough_points[["x", "y"]])
        self.topic_kdtree = spatial.KDTree(topic_points[["x", "y"]])
        self.rough_topic_points = topic_points

    """
    def add_bridges(self):
        graph = nx.read_gpickle(getFile(name, Datafile.GRAPH_PICKLE))
        # Basically, we want to examine each edge in the input graph
        # and try to see if there is an existing Voronoi ridge that
        # satisfies it.
        # If there isn't one, we look for a (water) Voronoi region
        # that touches the two regions we want to connect
        # and change its designation from water to land by adding it
        # to self.rough_topic_pts df

        dim = max(self.topic_df.index) + 1
        ridge_adjacency = np.zeros((dim, dim))
        for i, pair in enumerate(self.rough_voronoi.ridge_points):
            topicA = self.get_rough_topic(pair[0])
            topicB = self.get_rough_topic(pair[1])
            if topicA != -1 and topicB != -1:
                ridge_adjacency[topicA][topicB] = i
                ridge_adjacency[topicB][topicA] = i

        delaunay = spatial.Delaunay(self.rough_points)
        neighbors = defaultdict(set)
        for p in delaunay.simplices:
            for v1, v2 in itertools.combinations(p, 2):
                idx_i = self.get_rough_topic(v1)
                idx_j = self.get_rough_topic(v2)
                neighbors[v1].add(idx_j)
                neighbors[v2].add(idx_i)

        to_convert = []
        for edge in list(graph.edges()):
            # Check to see if there's already a border for this edge
            if edge[0] == 61 and edge[1] == 67:
                pdb.set_trace()
            if edge[0] == 67 and edge[1] == 61:
                pdb.set_trace()
            if ridge_adjacency[edge[0]][edge[1]]:
                continue
            # We couldn't find one.
            # Now, look for a V region that could link them.
            for n_ in neighbors:
                # topics is the set of neighboring topics of the voronoi polyon # N
                neighboring_topics = neighbors[n_]
                if edge[0] in neighboring_topics and edge[1] in neighboring_topics:
                    if n_ < len(self.rough_topic_points) or n_ in to_convert:
                        # Land polygon, we don't want to remove it (I think?)
                        continue
                    # TODO: choose source or target more intelligently
                    print(f"Adding point {n_} to connect {edge[0]} and {edge[1]}")
                    to_convert.append(n_)
                    self.rough_points.at[n_, "topic"] = edge[0]
                    break

        new_topic_points = pd.concat(
            (self.rough_topic_points, self.rough_points.iloc[to_convert]),
            ignore_index=True,
        )
        pdb.set_trace()
        mask = [
            i >= len(self.rough_topic_points) and i not in to_convert
            for i in range(len(self.rough_points))
        ]
        new_noise_pts = self.rough_points.iloc[mask]

        self.set_rough_points(new_topic_points, new_noise_pts)

        plt_voronoi(self.rough_points, self.topic_df)
    """

    def initial_elevation(self):
        # Assign topics to each hi-resolution polygon
        for i, simplex in enumerate(self.delaunay.simplices):
            pts = self.delaunay.points[simplex]
            pt = np.mean(pts, axis=0)
            _, tpt = self.rough_kdtree.query(pt)
            topic = int(self.rough_points["topic"][tpt])
            if topic == -1:
                # water
                self.elevation[i] = 0
            else:
                self.elevation[i] = self.topic_df["radius"][topic]

            try:
                """
                _, pt = self.topic_kdtree.query(pt)
                topic = int(self.rough_topic_points["topic"][pt])
                self.moisture[i] = self.topic_df["subjectivity"][topic]
                self.temperature[i] = self.topic_df["media_diversity"][topic]
                """
            except:
                pdb.set_trace()

    def smooth_coastline(self):
        # For nodes that have elevation == 0 and both neighbors elevation > 0, pull up
        # For nodes that have elevation > 0 and both neighbors elevation == 0, push down
        for i, neighbors in enumerate(self.delaunay.neighbors):
            nonedge_neighbors = [n for n in neighbors if n != -1]
            neighbor_elevation = self.elevation[nonedge_neighbors]
            threshold = len(nonedge_neighbors) / 2
            if self.elevation[i] < 0:
                if np.sum(neighbor_elevation > 0) > threshold:
                    self.elevation[i] = np.mean(neighbor_elevation)
            else:
                if np.sum(neighbor_elevation < 0) > threshold:
                    self.elevation[i] = np.mean(neighbor_elevation)

    def assign_topics(self):
        # Assign topics to each hi-resolution polygon
        for i, simplex in enumerate(self.delaunay.simplices):
            if self.elevation[i] > 0:
                pts = self.delaunay.points[simplex]
                pt = np.mean(pts, axis=0)
                dist, macroregion = self.topic_kdtree.query(pt)
                topic = int(self.rough_topic_points["topic"][macroregion])
                self.topics[i] = topic
            else:
                self.topics[i] = -1

    def __init__(self, n=30, name=""):
        """
        Initialize the PolygonMap class, with n points
        Arguments:
        n -- the number of points
        """
        # init healines to empty array of arrays

        # fill with n random 2D vectors
        self.name = name
        self.pts = np.random.random((n, 2))

        # use lloyd relaxation to space them better
        self.vor = spatial.Voronoi(self.pts)
        self.n_regions = len(self.vor.regions)
        self.pts = PolygonMap.improve_points(self.pts)
        # ok, so each voronoi vertex defines triangle, with the 3 points of the triangle =
        # the voronoi centers of the neighboring polygons
        self.delaunay = spatial.Delaunay(self.pts)
        self.n_triangles = len(self.delaunay.simplices)
        self.headlines = [[] for i in range(self.n_triangles)]
        self.topics = np.zeros(self.n_triangles)
        # use KD tree to provide nearest neighbor lookups for occasional grid based operation
        self.tree = spatial.KDTree(self.pts)

        # ridge points are pairs of points that have a polygon edge between them.
        # create a adjacency dictionary for the point indices
        # you could also call these adjacency regions tbh
        # so for each region, we can easily get a list of its neighbors
        for p0, p1 in self.vor.ridge_points:
            self.adj_points[p0].append(p1)
            self.adj_points[p1].append(p0)

        # do the same for the ridge vertices (the coordinates that define the edges)
        for v0, v1 in self.vor.ridge_vertices:
            self.adj_vertices[v0].append(v1)
            self.adj_vertices[v1].append(v0)

        # make a list of whether each region is on the edge or not
        self.edges = np.asarray(
            [-1 in neighbors for neighbors in self.delaunay.neighbors]
        )

        # make a list of each region's elevation. initialize to zero
        self.elevation = np.zeros(self.n_triangles)
        self.moisture = np.zeros(self.n_triangles)
        self.temperature = np.zeros(self.n_triangles)
        self.shadow = np.zeros(self.n_triangles)
        self.flux_map = np.zeros(self.n_triangles)

    @staticmethod
    def improve_points(pts, n=3):
        """
        Use Lloyd relaxation to improve the point spacing
        pts -- nparray of 2D points
        n -- number of times to use relaxation
        """
        for _ in range(n):
            vor = spatial.Voronoi(pts)
            newpts = []
            for idx in range(len(vor.points)):
                pt = vor.points[idx, :]
                region = vor.regions[vor.point_region[idx]]
                if -1 in region:
                    newpts.append(pt)
                else:
                    vxs = np.asarray([vor.vertices[i, :] for i in region])
                    vxs[vxs < 0] = 0
                    vxs[vxs > 1] = 1
                    newpt = np.mean(vxs, 0)
                    newpts.append(newpt)
            pts = np.asarray(newpts)
        return pts
        """
        for _ in range(n):
            vor = spatial.Voronoi(pts)
            newpts = []
            for i, pt in enumerate(vor.points):
                pt = pt.tolist()
                region = vor.regions[i]
                if not region or -1 in region:
                    # this region is on the outer edge, don't regularize it
                    newpts.append(pt)
                else:
                    # the region contains vertex indices,
                    # so pull them and take the mean
                    vxs = np.asarray([vor.vertices[i] for i in region])
                    newpt = np.mean(vxs, axis=0)
                    newpts.append(newpt.tolist())
            pts = newpts
        return np.asarray(pts)
        """

    def add_headline(self, headline, dominant_topic, x, y):
        # Add a headline with X, Y coordinate
        # This adds 1 to the elevation of the corresponding Voronoi region
        # _, point_i = self.tree.query([x, y])
        # _, point_i = self.manual_search(x, y)
        tri_i = self.delaunay.find_simplex((x, y))
        if tri_i == -1:
            print("Could not find simplex")
            return
        """
        if point_i >= len(self.vor.points):
            print(f"{headline} could not be placed")
            return
        region_i = self.vor.point_region[point_i]
        """

        self.elevation[tri_i] += 1
        self.headlines[tri_i].append(headline)

    def erode(self):
        # First get the planchon darboux:
        pd_heightmap = self.elevation.copy()
        # Fill everything that isn't an edge to infinity
        # self.pd_heightmap[self.edges == False] = np.inf
        pd_heightmap[self.edges != True] = np.inf
        epsilon = 0.00001
        print("calculate hillmap")
        while True:
            visited = np.zeros(len(pd_heightmap), dtype=bool)
            for i, height in enumerate(pd_heightmap):
                # If this cell has a neighbor lower than it, set this cell's
                # height to the max of its original height or the neighbor's height + epsilon

                # immediately continue if this is already at "normal" height
                if height == self.elevation[i]:
                    continue

                nonedge_neighbors = [n for n in self.delaunay.neighbors[i] if n != -1]
                lower_neighbor_heights = [
                    h for h in pd_heightmap[nonedge_neighbors] if h < pd_heightmap[i]
                ]
                # No lower neighbors
                if not lower_neighbor_heights:
                    continue
                new_val = np.max(
                    [np.max(lower_neighbor_heights) + epsilon, self.elevation[i]]
                )
                if pd_heightmap[i] != new_val:
                    pd_heightmap[i] = new_val
                    visited[i] = True
            if np.sum(visited) == 0:
                break

        print("calculate water flux")
        # Next, get the water flux map
        # Create a sorted list of nodes by height, keep track of the indexes
        pd_heightmap_sorted = [(i, val) for i, val in enumerate(pd_heightmap)]
        pd_heightmap_sorted.sort(key=lambda x: x[1], reverse=True)
        self.flux_map = np.array([1 for _ in range(len(pd_heightmap))])
        for node in pd_heightmap_sorted:
            nonedge_neighbors = [n for n in self.delaunay.neighbors[node[0]] if n != -1]
            neighbor_heights = pd_heightmap[nonedge_neighbors]
            lowest_neighbor = nonedge_neighbors[np.argmin(neighbor_heights)]
            # Give the lowest neighbor this node's rainfall
            self.flux_map[lowest_neighbor] += self.flux_map[node[0]]

        print("e r o d e")
        for i, flux in enumerate(self.flux_map):
            nonedge_neighbors = [n for n in self.delaunay.neighbors[node[0]] if n != -1]
            neighbor_slopes = np.abs(
                pd_heightmap[nonedge_neighbors] - self.elevation[i]
            )
            slope = np.mean(neighbor_slopes)
            delta = np.min([flux ** 0.5 * slope / 100, 0.1])
            self.elevation[i] -= delta
        # self.normalize()

    def normalize(self):
        """
        Normalize the heightmap to 0-1
        """
        self.elevation -= np.min(self.elevation)
        self.elevation /= np.max(self.elevation)

    def round_hills(self):
        """
        Round hills by applying square root function
        """
        assert np.min(self.elevation) >= 0
        assert np.min(self.elevation) <= 1
        assert np.max(self.elevation) <= 1
        assert np.max(self.elevation) >= 0
        self.elevation **= 0.5

    def smooth(self, n=2):
        """
        Average each triangle with its neighbors
        """
        for i in range(n):
            new_elevations = self.elevation.copy()
            new_moistures = self.moisture.copy()
            new_temp = self.temperature.copy()
            for i, neighbors in enumerate(self.delaunay.neighbors):
                neighbors = neighbors[neighbors != -1]
                new_elevations[i] = np.mean(
                    np.append(self.elevation[neighbors], self.elevation[i])
                )
                new_moistures[i] = np.mean(
                    np.append(self.moisture[neighbors], self.moisture[i])
                )
                new_temp[i] = np.mean(
                    np.append(self.temperature[neighbors], self.temperature[i])
                )

            self.elevation = new_elevations
            self.moisture = new_moistures
            self.temperature = new_temp

    def reset_sea_level(self, quantile=0.25):
        """
        Trim all of the bottom N landmass
        """
        self.elevation -= np.quantile(self.elevation, quantile)

    def calculate_shadows(self):
        sun = np.array([0, 0, 20])
        print("calculating shadows")
        for i, simplex in enumerate(self.delaunay.simplices):
            pts = self.delaunay.points[simplex]
            pt = np.mean(pts, axis=0)
            pt = np.array([pt[0], pt[1], self.elevation[i]])
            if i % 100 == 0:
                print(i, self.n_triangles, end="\r")
            delta = pt - sun
            orig = sun + 1 / delta[2] * delta
            LEN = 100
            j = 0
            while j < LEN:
                ray = orig + (pt - orig) * j / LEN
                tri_i = self.delaunay.find_simplex((ray[0], ray[1]))
                if ray[2] < self.elevation[tri_i]:
                    self.shadow[i] = 1
                    self.shadow[tri_i] = -1
                    break
                j += 1

    def export(self):
        """
        Export to a format that we can pass to JS frontend
        Honestly, the headlines might be slower to fetch
        We might need to build a backend LOL
        We'll just dump the points, edges, and elevations basically
        """
        df = pd.DataFrame(self.pts, columns=["x", "y"])
        # df.to_csv(getFile(self.name, Datafile.POINTS_TSV), sep="\t")
        # df = pd.DataFrame(self.vor.vertices, columns=["x", "y"])
        df.to_csv(getFile(self.name, Datafile.VERTICES_TSV), sep="\t")

        df = pd.DataFrame()
        df["elevation"] = self.elevation
        # df["coordinates"] = self.vor.regions
        df["coordinates"] = [
            json.dumps(row.tolist()) for row in self.delaunay.simplices
        ]
        df["headlines"] = self.headlines
        df["is_edge"] = self.edges
        # ugly but i hate np
        df["topics"] = self.topics
        df["flux"] = self.flux_map
        df["moisture"] = self.moisture
        df["temperature"] = self.temperature
        df["shadow"] = self.shadow

        df.to_csv(getFile(self.name, Datafile.REGIONS_TSV), sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn TSNE points into map data")
    parser.add_argument("-name", dest="name", required=True, help="name for input data")
    parser.add_argument(
        "-start",
        dest="start",
        help="start date (ISO)",
    )
    parser.add_argument(
        "-interval",
        dest="interval",
        default=28,
        type=int,
        help="Number of days to include after the given start date",
    )
    parser.add_argument(
        "-n",
        dest="n",
        required=False,
        default=1024,
        type=int,
        help="number of polygons to create",
    )
    parser.add_argument(
        "-group",
        dest="group",
        type=int,
        nargs="+",
        required=False,
        help="if provided, a group number to restrict to (useful for debugging)",
    )
    parser.add_argument(
        "-matplotlib",
        dest="matplotlib",
        action="store_const",
        const=True,
        required=False,
        help="if provided, output intermediate matplotlibs",
    )
    args = parser.parse_args()
    name = names.getName(args.name, args.start, args.interval)

    topic_df = pd.read_csv(
        getFile(name, Datafile.TOPIC_METADATA_TSV), sep="\t", index_col=0
    )

    with open(getFile(name, Datafile.LAYOUT), "rt") as f:
        layout = json.load(f)["layouts"]
        layout = sorted([item for sl in layout for item in sl], key=lambda x: x["id"])
        layout_df = pd.DataFrame(layout)
        layout_df.index = layout_df["id"]
    # this should be moved somewhere else :P
    assert (topic_df.index == layout_df.index).all()
    topic_df["x"] = layout_df["x"]
    topic_df["y"] = layout_df["y"]
    topic_df["group"] = layout_df["group"]

    if args.group is not None:
        topic_df = topic_df[topic_df["group"].isin(args.group)]

    # topic_df = topic_df.reset_index()
    # Now, topic #s (except assigned at the end in export, are the col indexes)

    # scale the x and y appropriately. Must be scaled together, not separately!
    xmin = np.min(topic_df["x"])
    xmax = np.max(topic_df["x"])

    ymin = np.min(topic_df["y"])
    ymax = np.max(topic_df["y"])

    largest = max(xmax - xmin, ymax - ymin)
    topic_df["x"] -= xmin
    topic_df["y"] -= ymin
    topic_df["x"] /= largest
    topic_df["y"] /= largest
    topic_df["radius"] = get_radius(topic_df)
    topic_df["radius"] /= largest
    print(largest)

    m = PolygonMap(args.n, name)
    m.create_initial_polygons(topic_df)
    m.initial_elevation()
    m.normalize()
    # m.smooth(n=3)
    # m.round_hills()
    # m.erode()
    # m.reset_sea_level(quantile=0.8)
    # Km.smooth_coastline()
    m.smooth_coastline()
    m.smooth_coastline()
    m.assign_topics()
    # m.calculate_shadows()
    m.export()
