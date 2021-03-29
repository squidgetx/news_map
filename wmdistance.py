"""
wmdistance class, optimized for finding multiple WMD distances on the same
distribution space and on BOW representations instead of sentences
"""
import logging
import numpy as np
import time
from pyemd import emd

logger = logging.getLogger(__name__)


class WMDistance:
    def __init__(self, model, dictionary, token_id_whitelist=None):
        self.model = model
        self.dictionary = dictionary
        self.token_ids = self.get_token_ids(token_id_whitelist)
        self.vocab_len = len(self.token_ids)
        self.distance_matrix = self.get_distance_matrix()

    def get_token_ids(self, token_id_whitelist):
        """
        return an array of token IDs representing the tokens in
        the dictionary that are also in the word vector model
        """
        # Remove out-of-vocabulary words.
        old_vocab_len = len(self.dictionary)
        if token_id_whitelist:
            token_ids = [
                tid
                for tid, token in self.dictionary.iteritems()
                if tid in token_id_whitelist and token in self.model
            ]
        else:
            token_ids = [
                tid for tid, token in self.dictionary.iteritems() if token in self.model
            ]
        vocab_len = len(token_ids)
        if vocab_len != old_vocab_len:
            print("Removed words from input vocab:", old_vocab_len - vocab_len)
            print("Vocab size:", vocab_len)
        return token_ids

    def get_distance_matrix(self):
        """
        return a distance matrix containing the w2v distance of every pair of words in
        the input vocabulary
        """
        # Compute distance matrix.

        distance_matrix = np.zeros((self.vocab_len, self.vocab_len))

        start = time.time()
        proc = 0
        print(f"calculating {self.vocab_len ** 2} distances")
        # If the token isn't in either document then we don't need to calculate the distance
        for i, t1 in enumerate(self.token_ids):
            for j, t2 in enumerate(self.token_ids):
                proc += 1
                if distance_matrix[i, j] != 0.0:
                    continue

                v1 = self.dictionary.id2token[t1]
                v2 = self.dictionary.id2token[t2]
                # Compute Euclidean distance between unit-normed word vectors.

                if proc % 100000 == 0:
                    print(
                        f"{int(proc / self.vocab_len ** 2 * 100)}% {time.time() - start}",
                        end="\r",
                    )
                distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                    np.sum(
                        (
                            self.model.get_vector(v1, norm=True)
                            - self.model.get_vector(v2, norm=True)
                        )
                        ** 2
                    )
                )
        end = time.time()
        print(f"total time for building distance matrix: {end - start}")
        return distance_matrix

    def get(self, document1, document2):
        """
        given bow representation (1d array of len len(dictionary)) of document, get WMD distance
        """
        assert np.sum(self.distance_matrix) > 0.0
        d1 = document1[self.token_ids]
        d2 = document2[self.token_ids]
        # Compute WMD.
        return emd(d1 / np.sum(d1), d2 / np.sum(d2), self.distance_matrix)


def wmdistance(model, document1, document2, dictionary):
    """Compute the Word Mover's Distance between two documents.
    When using this code, please consider citing the following papers:
    * `Ofir Pele and Michael Werman "A linear time histogram metric for improved SIFT matching"
        <http://www.cs.huji.ac.il/~werman/Papers/ECCV2008.pdf>`_
    * `Ofir Pele and Michael Werman "Fast and robust earth mover's distances"
        <https://ieeexplore.ieee.org/document/5459199/>`_
    * `Matt Kusner et al. "From Word Embeddings To Document Distances"
        <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.
    Parameters
    ----------
    model : w2v model
    document1 : BOW representation of document 1 (1D np.array<float> of length vocab_size)
        Input document.
    document2 : BOW representation of document 2 (1D np.array<float> of length vocab_size)
        Input document.
    dictionary : gensim dictionary to provide id2token mapping for each document

    Returns
    -------
    float
        Word Mover's distance between `document1` and `document2`.
    Warnings
    --------
    This method only works if `pyemd <https://pypi.org/project/pyemd/>`_ is installed.
    If one of the documents have no words that exist in the vocab, `float('inf')` (i.e. infinity)
    will be returned.
    Raises
    ------
    ImportError
        If `pyemd <https://pypi.org/project/pyemd/>`_  isn't installed.
    """

    # If pyemd C extension is available, import it.
    # If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance

    # Remove out-of-vocabulary words.
    old_vocab_len = len(document1)
    token_ids = [tid for tid, token in dictionary.iteritems() if token in model]
    vocab_len = len(token_ids)
    if vocab_len != old_vocab_len:
        logger.info("Removed %d OOV words from vocab", old_vocab_len - vocab_len)
    document1 = document1[token_ids]
    document2 = document2[token_ids]

    if not document1.any() or not document2.any():
        logger.info(
            "At least one of the documents had no words that were in the vocabulary. "
            "Aborting (returning inf)."
        )
        return float("inf")

    if vocab_len == 1:
        # Both documents are composed by a single unique token
        return 0.0

    # Sets for faster look-up.
    docset = set(token_ids)

    # Compute distance matrix.
    distance_matrix = np.zeros((vocab_len, vocab_len))

    start = time.time()
    proc = 0
    print(f"calculating {vocab_len ** 2} distances")
    # If the token isn't in either document then we don't need to calculate the distance
    for i, t1 in enumerate(token_ids):
        if document1[i] == 0:
            continue

        for j, t2 in enumerate(token_ids):
            proc += 1
            if document2[i] == 0 or distance_matrix[i, j] != 0.0:
                continue

            v1 = dictionary.id2token[t1]
            v2 = dictionary.id2token[t2]
            # Compute Euclidean distance between unit-normed word vectors.

            if proc % 100000 == 0:
                print(
                    f"{int(proc / vocab_len ** 2 * 100)}% {time.time() - start}",
                    end="\r",
                )
            distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(
                np.sum(
                    (model.get_vector(v1, norm=True) - model.get_vector(v2, norm=True))
                    ** 2
                )
            )
    end = time.time()
    print(f"total time for building distance matrix: {end - start}")

    if np.sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        logger.info("The distance matrix is all zeros. Aborting (returning inf).")
        return float("inf")

    # Compute WMD.
    return emd(
        document1 / np.sum(document1), document2 / np.sum(document2), distance_matrix
    )
