from waybacknews.searchapi import SearchApiClient
import datetime as dt
import csv
import pdb
import argparse
import names

api = SearchApiClient("mediacloud")

NAME = "us-mainstream-stories"
# From the mediacloude top newspapers 2018 collection
websites = [
    "nytimes.com",
    "washingtonpost.com",
    "usatoday.com",
    "latimes.com",
    "nypost.com",
    "nydailynews.com",
    "sfgate.com",
    "bostonglobe.com",
    "reuters.com",
    "foxnews.com",
    "cnn.com",
    "newsweek.com",
    "forbes.com",
    "msn.com",
    "wsj.com",
    "dailymail.co.uk",
    "telegraph.co.uk",
    "theguardian.com",
    "cbsnews.com",
    "cnet.com",
    "time.com",
    "huffingtonpost.com",
]

arg_parser = argparse.ArgumentParser(
    description="Fetch headlines from Internet Archive Wayback collection. If no arguments are supplied, attempts to refresh the DB to today using metadata.json"
)
arg_parser.add_argument(
    "-start",
    dest="start",
    required=True,
    help="ISO formatted start date for query (ex: 2020-01-01)",
)
arg_parser.add_argument(
    "-end",
    dest="end",
    required=False,
    help="ISO formatted end date for query (ex: 2020-01-31). Defaults to today.",
)
arg_parser.add_argument(
    "-keyword", dest="keyword", required=False, help="keyword to search for"
)

args = arg_parser.parse_args()
start_dt = dt.datetime.fromisoformat(args.start)
end_dt = dt.datetime.fromisoformat(args.end)
interval = (start_dt - end_dt).days

records = []
for website in websites:
    print(website)
    for page in api.all_articles(f"domain:{website}", start_dt, end_dt):
        for a in page:
            if a["language"] != "en":
                continue
            records.append(
                {
                    "title": a["title"],
                    "url": a["url"],
                    "publish_date": a["publication_date"],
                    "domain": a["domain"],
                }
            )
print(len(records))

name = names.getName(NAME, args.end, interval)
filename = names.getFile(name, names.Datafile.HEADLINES_TSV)
with open(filename, "w") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=records[0].keys(), delimiter="\t")
    writer.writeheader()
    writer.writerows(records)
