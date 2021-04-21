# Python script to take a tsv of coordinates as input and output map
# Heavily inspired by https://github.com/mewo2/deserts

import numpy as np
import json
import pdb
import logging
import argparse
import pandas as pd
from time import sleep
import numpy as np
import scipy.spatial as spatial
from scipy.spatial import KDTree
import scipy.stats as stats
import sys
from names import getFile, Datafile
import names
from collections import defaultdict


class PolygonMap:
    # invariant: all points between 0-1
    adj_points = defaultdict(list)
    adj_vertices = defaultdict(list)

    headlines = []

    def __init__(self, n=30, name=""):
        """
        Initialize the PolygonMap class, with n points
        Arguments:
        n -- the number of points
        """
        # init healines to empty array of arrays

        # fill with n random 2D vectors
        self.pts = np.random.random((n, 2))
        self.name = name

        # use lloyd relaxation to space them better
        self.pts = PolygonMap.improve_points(self.pts)
        self.vor = spatial.Voronoi(self.pts)
        self.n_regions = len(self.vor.regions)
        # ok, so each voronoi vertex defines triangle, with the 3 points of the triangle =
        # the voronoi centers of the neighboring polygons
        self.delaunay = spatial.Delaunay(self.pts)
        self.n_triangles = len(self.delaunay.simplices)
        self.headlines = [[] for i in range(self.n_triangles)]
        self.topics = [[] for i in range(self.n_triangles)]
        # use KD tree to provide nearest neighbor lookups for occasional grid based operation
        self.tree = KDTree(self.pts)

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
        self.topics[tri_i].append(dominant_topic)

    def normalize(self):
        """
        Normalize the heightmap to 0-1
        """
        self.elevation /= np.linalg.norm(self.elevation)

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
            new_topics = self.topics.copy()
            for i, neighbors in enumerate(self.delaunay.neighbors):
                neighbors = neighbors[neighbors != -1]
                new_elevations[i] = np.mean(
                    np.append(self.elevation[neighbors], self.elevation[i])
                )
                if self.topics[i] == []:
                    for n in neighbors:
                        self.topics[i].extend(self.topics[n])

            self.elevation = new_elevations

    def reset_sea_level(self, quantile=0.25):
        """
        Trim all of the bottom N landmass
        """
        self.elevation[self.elevation < np.quantile(self.elevation, quantile)] = 0

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
        df["topics"] = [stats.mode(a).mode for a in self.topics]
        df["topics"] = [a[0] if a.size > 0 else [-1] for a in df["topics"]]

        df.to_csv(getFile(self.name, Datafile.REGIONS_TSV), sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn TSNE points into map data")
    parser.add_argument("-name", dest="name", required=True, help="name for input data")
    parser.add_argument(
        "-n",
        dest="n",
        required=False,
        default=1024,
        type=int,
        help="number of polygons to create",
    )
    args = parser.parse_args()

    m = PolygonMap(args.n, args.name)
    scores = np.genfromtxt(
        getFile(args.name, Datafile.RUST_PROBABILITIES), delimiter=","
    )
    headlines = pd.read_csv(getFile(args.name, Datafile.HEADLINES_TSV), sep="\t")
    sizes = names.getTopicSizes(args.name)
    with open(getFile(args.name, Datafile.LAYOUT), "rt") as f:
        layout = json.load(f)["layouts"]
        layout = sorted([item for sl in layout for item in sl], key=lambda x: x["id"])
        layout_df = pd.DataFrame(layout)

    # scale the x and y appropriately. Must be scaled together, not separately!
    xmin = np.min(layout_df["x"])
    xmax = np.max(layout_df["x"])

    ymin = np.min(layout_df["y"])
    ymax = np.max(layout_df["y"])

    largest = max(xmax - xmin, ymax - ymin)
    layout_df["x"] -= xmin
    layout_df["y"] -= ymin
    layout_df["x"] /= largest
    layout_df["y"] /= largest

    for index, row in enumerate(scores):
        topic = int(np.argmax(row))
        x = layout_df.iloc[topic]["x"]
        y = layout_df.iloc[topic]["y"]
        # dx = (layout_df["x"] - x) * row
        # dy = (layout_df["y"] - y) * row
        # x += dx.sum()
        # y += dy.sum()
        """
        for i, val in enumerate(row):
            if val < 0.01:
                continue
            if i == topic:
                continue
            # for each topic, adjust this headline toward based on the strength of its probability
            # most of the time, there shouldn't actually be much shifting I think
            x1 = layout_df.iloc[i]["x"]
            y1 = layout_df.iloc[i]["y"]
            x += (x1 - x0) * val
            y += (y1 - y0) * val
        """
        if index % 1000 == 0:
            print(index, len(scores), end="\r")

        factor = (sizes[topic] ** 0.5) / 2000
        x += np.random.normal(0, (1 - row.max()) * factor)
        y += np.random.normal(0, (1 - row.max()) * factor)
        m.add_headline(headlines["title"][index], topic, x, y)
    m.normalize()
    m.round_hills()
    m.smooth()
    m.reset_sea_level(quantile=0.5)
    m.export()
