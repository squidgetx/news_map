# Setup

Install python and node dependencies:

```
pip3 install -r requirements.txt
npm i
```

Build the gsdmm-rust binary:

```
cd gsdmm-rust && cargo build --release
```

python:

```
import nltk
nltk.download('stopwords')
```

# Making a new map

## Download the data

```
python get_stories_wayback --start <st> --end <et>
```

## Clean and prep the data

```
python splitter.py <filename>
```

## Train a new topic model

```
python get_topics.py -name us-mainstream-stories -start <st>
```

## Build a layout

```
node layout.js -name us-mainstream-stories_NAME
```

## Build the map

```
python topic2map.py -name -start -i
```

## Test the map

update src/files.js and src/main.js to use the new dates, run python -m http.server and localhost:8000/dist shows the map. click and drag labels

# Build

In order to update the build site:

```
git subtree split --prefix dist master
<returns token>
git push origin <token>:gh-pages --force
```
