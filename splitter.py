"""
Script to split headlines downloaded from Mediacloud into daily files
Also does some basic cleaning and preprocessing
"""

import argparse
import pandas as pd

import get_topics
import names

def split(file):
    df = pd.read_csv(file, sep="\t", parse_dates=["publish_date"])
    df = get_topics.clean(df)

    df["publish_date_day"] = df["publish_date"].dt.date

    for i, group_ in df.groupby("publish_date_day"):
        print(f"wrote {str(i)}")
        group_.reset_index().to_csv(f"data/raw/{names.Basename.US_MAINSTREAM.value}_{str(i)}.tsv", sep="\t")

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Split a giant TSV into smaller ones by day for easier consumption"
    )

    arg_parser.add_argument("-file", dest="file", help="filename to split")

    args = arg_parser.parse_args()
    split(args.file)
  