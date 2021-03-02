# Python script to take a set of words as input and output sound
# The input eventually will be the output of a topic model 
# but for now we will just have sample input words

import numpy as np
import logging
import simplecoremidi
import pandas as pd
from time import sleep
import math
import sys

EMBEDDINGS = {}
TOPICS = []

def load_embeddings(stats=False):
    print("Loading embeddings...")
    with open("glove.6B/glove.6B.50d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            EMBEDDINGS[word] = vector
    if stats:
        df = pd.DataFrame(data=EMBEDDINGS).transpose()
        stats = []
        for i in range(50):
            stats.append({
                'mean': np.mean(df[i]),
                'min': np.min(df[i]),
                'max': np.max(df[i]),
                'var': np.var(df[i]),
            })
        print(pd.DataFrame(data=stats))
            
    print("Done!")
        # Show some descriptive stats for the embeddings:

    
def scale(old_value, old_min, old_max, new_min, new_max):
    return ((old_value - old_min) /
            (old_max - old_min)) * (new_max - new_min) + new_min

def get_dim(word, min_word, max_word, clip=True):
    vec = EMBEDDINGS[word]
    a = EMBEDDINGS[min_word]
    b = EMBEDDINGS[max_word]
    '''Project the vector onto the line defined by a, b 
    and return a value 0-1 where a=0, b=1'''
    ap = vec - a
    ab = b - a
    point = a + np.dot(ap,ab) / np.dot(ab,ab) * ab
    angle = np.arccos(np.dot(ap/np.linalg.norm(ap), ab/np.linalg.norm(ab)))
    mag = np.linalg.norm(point - a) / np.linalg.norm(b - a)

    ap = point - a
    dp = np.dot(ap/np.linalg.norm(ap), ab/np.linalg.norm(ab))
    if dp <= -0.9:
        # The vectors face opposite direction: point is less than A
        if clip:
            logging.warning(f"{word} less than {min_word}: {-mag}")
            mag = 0.0
        return -mag, angle
    else:
        # The vectors face same direction: point is greater than A
        if mag > 1:
            if clip:
                logging.warning(f"{word} greater than {max_word}: {mag}")
                mag = 1.0
        return mag, angle


def find_dims(min_word, max_word):
    a = EMBEDDINGS[min_word]
    b = EMBEDDINGS[max_word]
    thresh = math.pi / 4
    for wo in EMBEDDINGS:
        mag, angle = get_dim(wo, min_word, max_word, clip=False)
        if mag < 0 or mag > 1:
            if angle < thresh or math.pi - angle < thresh:
                print(mag, angle, wo)


def topic_to_sound(topic):
    if not EMBEDDINGS:
        load_embeddings(stats=False)

    
    root_note = 60  # This is middle C
    channel = 1  # This is MIDI channel 1
    note_on_action = 0x90
    control_action = 0xB0
    #find_dims('natural', 'artificial')
    while True:
        for word in topic:
            vec = EMBEDDINGS[word]
            org_syn, _ = get_dim(word, 'natural', 'artificial')
            print(word, org_syn)
            simplecoremidi.send_midi((control_action | channel, 0x01, int(org_syn * 127)))
            simplecoremidi.send_midi((note_on_action | channel, root_note, 100))
            sleep(1)
            simplecoremidi.send_midi((note_on_action | channel, root_note, 0))
            sleep(0.1)
    
    # A topic is some finite (and small-ish number of words) and associated weights
    # Our input is actually a series of topics: say, top N topics N=3 or something arbitrary

    # Is topic modeling actually required here or would it make more sense to sonify the 
    # input text directly?
    # + topic modeling standardizes input but loses some fidelity. 
    # + topic modeling allows multiple articles to be combined together more easily 
    # + topic modeling lets us more easily recognize similarity among stations
    # what would a non-topic modeling approach look like
    # I guess the topic modelling component is useful for standardization/generalization

    # ok, so what are our sonification strategies, knowing that multiple topics may be sonified at once 
    # with different weights?
    # 1. sum all words in the topic together to form a single wordvector
    # 2. oscillate between them proportionally or stochastically
    # 3. layer them all on top of each other (nested tracks) with proportional or stochastic volme level
    # Open question: what is the time domain here?
    # 4. geometric interpretation: we have a 50D shape 

    # let's shift gears to sound stuff then. many approaches here of course
    # Some basic criteria we could want:
    # If a topic is tightly clustered, then the music should be more "congruous"
    # If a topic is sparsely clustered, then the music could be more "incongruous"
    # Topic "center" should result in varying musical "topics"
    #   so this could relate to melody, genre, etc. 
    # What does it mean to have the same topic but transposed in the vector space?
    #   idk if this would really happen but it would be like "king actor" => "queen actress"
    #   it might be more complicated?

    # ok, so primary strategy could just be interpolation with a few fixed values idkk

    # thoughts 2/6
    # set up a certain number of dimensions and project words onto those dimensions
    # for each word you have a 



if __name__ == '__main__':
    args = sys.argv[1:]
    d = {w: 1 for w in args}
    topic_to_sound(d)



