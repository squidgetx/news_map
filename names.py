"""
utility file to manage filenames for consistency
"""
from enum import Enum


class Datafile(Enum):
    RUST_CLUSTER_DESC = "cluster_descriptions.txt"
    RUST_LABELS = "labels.csv"
    DICTIONARY = ".dictionary"
    TOPIC_NDARRAY = ".topic_ndarray.npy"
    TOPIC_JSON = ".topics.json"
    SCORES = ".scores.tsv"
    DISTANCE_JS = ".distanceJS.tsv"
    DISTANCE_WMD = ".distanceWMD.tsv"
    MDS = ".mds.tsv"
    TOPIC_ADJACENCY = ".topic_adjacency.json"


def getFile(name, suffix: Datafile):
    return f"data/{name}{suffix.value}"