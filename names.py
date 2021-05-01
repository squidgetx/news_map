"""
utility file to manage filenames for consistency
"""
from enum import Enum
import json
import pandas as pd
from datetime import datetime, timedelta


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
    GRAPH_PICKLE = ".graph.gpickle"
    LAYOUT = ".layout.json"
    REGIONS_TSV = ".regions.tsv"
    POINTS_TSV = ".points.tsv"
    VERTICES_TSV = ".vertices.tsv"
    INTERDISTANCE_JS = ".interdistance.tsv"
    HEADLINES_TSV = ".headlines.tsv"
    TOPIC_METADATA_TSV = ".topics.metadata.tsv"
    TOPIC_RECORDS = ".topic_records.json"
    METACLUSTERS = ".metaclusters.json"
    METAMETACLUSTERS = ".metametaclusters.json"
    FULL_GRAPH = ".fullgraph.json"


def getFile(name: str, suffix: Datafile):
    return f"data/{name}{suffix.value}"


def readJSON(name: str, suffix: Datafile):
    return json.load(open(getFile(name, suffix)))


def readTSV(name: str, suffix: Datafile):
    return pd.read_csv(getFile(name, suffix))


def getTopicSizes(name):
    # get the sizes
    topic_json = json.load(open(getFile(name, Datafile.TOPIC_JSON)))
    sizes = {}
    for topic in topic_json:
        sizes[int(topic)] = topic_json[topic]["size"]
    return sizes


def getName(basename: str, datestr: str, interval: int):
    start_date = datetime.fromisoformat(datestr)
    end_date = start_date + timedelta(days=interval)
    return basename + "_" + str(start_date.date()) + "_" + str(end_date.date())


def getPrevName(basename: str, datestr: str, interval: int, step: int):
    start_date = datetime.fromisoformat(datestr) - timedelta(days=step)
    end_date = datetime.fromisoformat(datestr) + timedelta(days=interval - step)
    return basename + "_" + str(start_date.date()) + "_" + str(end_date.date())


def getEndDateStr(datestr: int, interval: int):
    end_date = datetime.fromisoformat(datestr) + timedelta(days=interval)
    return str(end_date.date())
