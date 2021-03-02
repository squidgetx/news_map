# Python script to take an input corpus and output topics
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import itertools
from gensim import corpora
from gensim import models
from gensim.test.utils import datapath
import fileinput
import logging
import argparse
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


OTHER_STOPWORDS = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can', 'one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']
num_topics = 20

def get_dict_corpus(sentences):
    sentences = [word_tokenize(s) for s in sentences]
    # skip lemmatizing for now
    swords = stopwords.words('English') + OTHER_STOPWORDS
    all_words = list(itertools.chain.from_iterable(sentences))
    all_words = [ w for w in all_words if w.isalpha() and w.lower() not in swords and len(w) > 1 ]
    dictionary_LDA = corpora.Dictionary([all_words])
    corpus = [dictionary_LDA.doc2bow(sentence) for sentence in sentences]
    return dictionary_LDA, corpus


def sentences_to_topic_model(sentences):
    # first, clean:
    #dictionary_LDA.filter_extremes(no_below=2)
    dictionary, corpus = get_dict_corpus(sentences)
    model = models.LdaModel(corpus, num_topics=num_topics, 
                                  id2word=dictionary, 
                                  passes=4, alpha=[0.01]*num_topics, 
                                  eta=[0.01]*len(dictionary.keys()))

  
    return dictionary, corpus, model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Train LDA topic model on input text'
    )
    parser.add_argument('-save', dest='save', required=False,
                        help='filename for saved model file')
    parser.add_argument('-load', dest='load', required=False,
                        help='filename for saved model file')
    parser.add_argument('-train', dest='train', required=True,
                        help='filename for training')
    args = parser.parse_args()

    logging.info(f"Opening training file {args.train}")
    model = None
    dictionary = None
    corpus = None
    sentences = None
    with open(args.train, 'rt') as f:
        sentences = [line for line in f]
    if args.load:
        logging.info(f"Loading saved model file {args.load}")
        model = models.LdaModel.load(datapath(args.load))
        dictionary, corpus = get_dict_corpus(sentences)
    else:
        logging.info(f"Training model...")
        dictionary, corpus, model = sentences_to_topic_model(sentences)

    if args.save:
        logging.info(f"Model saved to {args.save}")
        lda.save(datapath(args.save))
    
    if not model:
        logging.error("Model failed to load")
        exit()

    topic_scores = [model[c] for c in corpus]
    with open('topics.tsv', 'wt') as f:
        f.write('n\ttopic\n')
        for i,topic in model.show_topics(formatted=True, num_topics=num_topics, num_words=20):
            f.write(str(i)+"\t"+ topic)
            f.write('\n')
    with open('output.tsv', 'wt') as f:
        f.write('headline\ttopic\tprobability\n')
        for i,t in enumerate(topic_scores):
            for score in t:
                f.write(f"{sentences[i]}\t{score[0]}\t{score[1]}\n")

    topic_weights = []
    for weights in topic_scores: 
        topic_weights.append({w[0]: w[1] for w in weights})

    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    #arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)
    with open('tsne.tsv', 'wt') as f:
        f.write('headline\tx\ty\tt\n')
        for i, t in enumerate(tsne_lda):
            f.write(f"{sentences[i].strip()}\t{t[0]}\t{t[1]}\t{topic_num[i]}\n")
            # bug? dup rows?


    
