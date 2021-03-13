# Python script to take a tsv of coordinates as input and output map
# Heavily inspired by https://github.com/mewo2/deserts

import numpy as np
import logging
import argparse
import pandas as pd
from time import sleep
import numpy as np
import scipy.spatial as spatial
from scipy.spatial import KDTree 
import sys
from collections import defaultdict


class PolygonMap:
    # invariant: all points between 0-1
    adj_points = defaultdict(list)
    adj_vertices = defaultdict(list)

    headlines = []

    def __init__(self, n=30):
        """
        Initialize the PolygonMap class, with n points
        Arguments:
        n -- the number of points
        """ 
        # init healines to empty array of arrays
        self.headlines = [[] for i in range(n+1)]

        # fill with n random 2D vectors
        self.pts = np.random.random((n, 2))

        # use lloyd relaxation to space them better
        self.pts = PolygonMap.improve_points(self.pts)
        self.vor = spatial.Voronoi(self.pts)

        # use KT tree to provide nearest neighbor lookups for occasional grid based operation
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
        self.edges = np.asarray([-1 in region for region in self.vor.regions])

        # make a list of each region's elevation. initialize to zero
        self.elevation = np.zeros(n + 1)

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

    def add_headline(self, headline, x, y):
        # Add a headline with X, Y coordinate
        # This adds 1 to the elevation of the corresponding Voronoi region
        _, region_i = self.tree.query([x, y])
        self.elevation[region_i] += 1
        self.headlines[region_i].append(headline)
    
    def export(self):
        """
        Export to a format that we can pass to JS frontend
        Honestly, the headlines might be slower to fetch 
        We might need to build a backend LOL
        We'll just dump the points, edges, and elevations basically
        """
        df = pd.DataFrame(self.vor.vertices, columns=['x', 'y'])
        df.to_csv('vertices.tsv', sep='\t')

        df = pd.DataFrame()
        df['elevation'] = self.elevation
        df['coordinates'] = self.vor.regions
        df['headlines'] = self.headlines
        df['is_edge'] = self.edges

        df.to_csv('regions.tsv', sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Turn TSNE points into map data'
    )
    parser.add_argument('-filename', dest='filename', required=True,
                        help='filename for input tsne data')
    args = parser.parse_args()

    m = PolygonMap(128)
    df = pd.read_csv(args.filename, sep='\t')
    # scale the x and y appropriately. Must be scaled together, not separately!
    xmin = np.min(df['x'])
    xmax = np.max(df['x'])

    ymin = np.min(df['y'])
    ymax = np.max(df['y'])
    
    largest = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    df['x'] += largest
    df['y'] += largest
    df['x'] /= largest * 2
    df['y'] /= largest * 2

    for index, row in df.iterrows():
        m.add_headline(row['headline'], row['x'], row['y'])
    m.export()
