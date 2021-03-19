'''
Script to fetch stories from Mediacloud and store the headlines in a tsv file
'''

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

US_NATIONAL_TAG=34412234
US_MAINSTREAM=8875027
BATCH_SIZE = 1000
SAVE_INTERVAL = 20000
TIMEOUT = 60
TIMEOUT_ERROR = 'request timed out'

METADATA_FILENAME = 'metadata.json'
DATA_FILENAME = 'stories.tsv'

class MediaCloudTimeoutException(Exception):
    pass

def handler(signum, frame):
    # Timeout triggered!
    print("Time ran out!")
    raise MediaCloudTimeoutException(TIMEOUT_ERROR)

# It looks like we might need to build something to re-start when media cloud is being angry at us
# ugh
# Also, we should build a daily fetcher along with a bulk fetcher
# Basically, default mode should be to "continue" from whatever data was last fetched
# Metadata can indicate error, meaning previous query wasn't successful, (needs retry)
# or, metadata can indicate success (but last refresh data was stale), so we need to fetch new data
# or, metadata can indicate success and last refresh data wasn't stale, so we don't need to do anything
def get_continue_params(metadata):
    error = metadata.get('error')
    if error:
        # we need to redo the query from where we left off
        return metadata['last_query']

    ds = metadata.get('latest')
    if ds and ds == str(datetime.date.today()):
        # we're fine don't do anything
        return None
    
    # parse starting from last to today
    return {
        'start': ds,
        'end': str(datetime.date.today()),
        'last_processed_id': 0,
    }

def save(stories, query, error=False, exc=""):
    if error:
        print(f"Ran into error!")
    print(f"fetched {len(stories)} total!")
    if stories:
        query['last_processed_id'] = stories[-1]['processed_stories_id']
    else:
        query['last_processed_id'] = 0

    df = pd.DataFrame(stories)

    # chuck the whole thing into a df
    # append without headers if the file exists already, otherwise create a new file
    if os.path.exists(DATA_FILENAME):
        with open(DATA_FILENAME, 'at') as f:
            df.to_csv(f, sep='\t', header=False)
    else:
        with open(DATA_FILENAME, 'wt') as f:
            df.to_csv(f, sep='\t', header=True)
    
    print(f"Saved to {DATA_FILENAME}")

    # always rescan to get the latest date
    #df_all = pd.read_csv(DATA_FILENAME, sep='\t')
    latest_date = str(np.max(pd.to_datetime(df['publish_date'])).date())
    new_metadata = {
        'error': error,
        'exc': exc,
        'last_query': query,
        'latest': latest_date
    }

    with open(METADATA_FILENAME, 'wt') as f:
        f.write(json.dumps(new_metadata))
    print(f"Metadata saved to {METADATA_FILENAME}")


if __name__=='__main__':

    arg_parser = argparse.ArgumentParser(
        description='Fetch headlines from Mediacloud DB. If no arguments are supplied, attempts to refresh the DB to today using metadata.json'
    )
    arg_parser.add_argument('-start', dest='start', required=False,
                        help='ISO formatted start date for query (ex: 2020-01-01). Defaults to latest value in metadata JSON.')
    arg_parser.add_argument('-end', dest='end', required=False,
                        help='ISO formatted end date for query (ex: 2020-01-31). Defaults to today.')
    args = arg_parser.parse_args()

    query = {}
    metadata = {}
    if os.path.exists(METADATA_FILENAME):
        with open(METADATA_FILENAME, 'rt') as f:
            metadata = json.load(f)

    query = get_continue_params(metadata)
    if not query:
        print("Latest refresh was today. No more work to do!")
        sys.exit()


    if args.start:
        query['start'] = args.start

    if args.end:
        query['end'] = args.end
    
    if not query:
        print("Unable to determine start and end dates.")
    
    start = parser.parse(query['start'])
    end = parser.parse(query['end'])
    
    mc = mediacloud.api.MediaCloud(os.environ.get('MEDIACLOUD_API_KEY'))
    stories = []
    error = False
    last_processed_stories_id = query['last_processed_id']
    while True: 
        print(f"fetching story {len(stories)} through {len(stories) + BATCH_SIZE}...")
        # register the handler
        signal.signal(signal.SIGALRM, handler)
        # set timer
        signal.alarm(TIMEOUT)
        try:
            fetched_stories = mc.storyList(
                f'coronavirus AND tags_id_media:{US_MAINSTREAM}',
                # default to today
                solr_filter=mc.dates_as_query_clause(start, end), 
                last_processed_stories_id=last_processed_stories_id, rows= BATCH_SIZE)
        except MediaCloudTimeoutException as exc:
            save(stories, query, error=True, exc=exc)
            break
        # cancel the alarm
        signal.alarm(0)
        stories.extend(fetched_stories)
        if len(fetched_stories) < BATCH_SIZE:
            # Didn't fetch the max amount, we're good
            break
        last_processed_stories_id = stories[-1]['processed_stories_id']
        if len(stories) >= SAVE_INTERVAL:
            save(stories, query)
            stories = []

    save(stories, query)
