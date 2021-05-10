"""
Script to fetch stories from Mediacloud and store the headlines in a tsv file
"""

from dateutil import parser
import mediacloud.api
import os
import sys
import datetime
import json
import np
import argparse
import signal
import pandas as pd
import get_topics

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Split a giant TSV into smaller ones by day for easier consumption"
    )

    arg_parser.add_argument("-file", dest="file", help="filename to split")
    arg_parser.add_argument("-name", dest="name", help="filename to output")

    args = arg_parser.parse_args()
    df = pd.read_csv(
        args.file, sep="\t", parse_dates=["publish_date"], infer_datetime_format=True
    )
    df = get_topics.clean(df)

    df["publish_date_day"] = df["publish_date"].dt.date

    for i, group_ in df.groupby("publish_date_day"):
        print(f"wrote {str(i)}")
        group_.reset_index().to_csv(f"data/raw/{args.name}_{str(i)}.tsv", sep="\t")
