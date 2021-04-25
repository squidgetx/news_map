"""
utility file to manage filenames for consistency
"""
from enum import Enum
import json


class Datafile(Enum):
    RUST_CLUSTER_DESC = "cluster_descriptions.txt"
    RUST_LABELS = "labels.csv"
    RUST_PROBABILITIES = "label_probabilities.csv"
    DICTIONARY = ".dictionary"
    TOPIC_NDARRAY = ".topic_ndarray.npy"
    TOPIC_JSON = ".topics.json"
    MEDIA_TOPIC_JSON = ".mediatopics.json"
    SCORES = ".scores.tsv"
    DISTANCE_JS = ".distanceJS.tsv"
    DISTANCE_WMD = ".distanceWMD.tsv"
    MDS = ".mds.tsv"
    TOPIC_ADJACENCY = ".topic_adjacency.json"
    LAYOUT = ".layout.json"
    REGIONS_TSV = ".regions.tsv"
    POINTS_TSV = ".points.tsv"
    VERTICES_TSV = ".vertices.tsv"
    INTERDISTANCE_JS = ".interdistance.tsv"
    HEADLINES_TSV = ".headlines.tsv"
    TOPIC_METADATA_TSV = ".topics.metadata.tsv"
    METACLUSTERS = ".metaclusters.json"
    METAMETACLUSTERS = ".metametaclusters.json"


def getFile(name: str, suffix: Datafile):
    return f"data/{name}{suffix.value}"


def getTopicSizes(name):
    # get the sizes
    topic_json = json.load(open(getFile(name, Datafile.TOPIC_JSON)))
    sizes = {}
    for i, topic in enumerate(topic_json):
        sizes[i] = topic["_metadata_"]["total"]
    return sizes
