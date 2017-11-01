import random
from gensim.models import Word2Vec
from nltk.corpus import brown
import numpy as np
import pandas as pd
import csv
from timeit import default_timer as timer

from scipy import spatial
# import nltk
# nltk.download('words')

def main():
    f = open("input.txt", "r")
    sentence_list = f.readlines()
    f.close()

    # Remove new line character from end of every sentence
    for i in range(len(sentence_list)):
        sentence_list[i] = sentence_list[i][:-1]
    print(sentence_list)

    glove_file_name = "glove.840B.300d" # This one takes forever, load it once then use pickel!!!
    # glove_file_name = "glove.6B.200d"
    words = loadGloveModel(glove_file_name)
    start = timer()
    pd.read_pickle("pickel_files/{}.pkl".format(glove_file_name))
    stop = timer()
    minutes = (stop - start) // 60
    secs = (stop - start) % 60
    print("Read pickel took {} mins {} secs".format(minutes, secs))

    word_pairs = randomlySelectWordPairs(sentence_list)
    calcCosSimilarity(words, word_pairs)


def randomlySelectWordPairs(sentence_list, n=25):
    '''
    Returns a list of word pairs, list of length n, pairs found in sentences of sentence_list
    :param sentence_list: List of sentences
    :param n: Number word pairs desired (Optional, default = 25)
    :return: List of word pairs
    '''
    sentence_list_for_pairs = []
    for i in range(len(sentence_list)):
        if len(sentence_list[i].split()) > 1:
            sentence_list_for_pairs.append(sentence_list[i])


    word_pairs = []
    for i in range(n):
        sentence = random.choice(sentence_list_for_pairs)
        word_list = sentence.split()
        randint = random.randrange(0,len(word_list)-1)
        word_pair = word_list[randint] + " "
        word_pair += word_list[randint+1]
        word_pairs.append(word_pair)

    print(word_pairs)

    return word_pairs

def calcCosSimilarity(words, word_pairs):
    errors = "ERRORS:\n"
    for pair in word_pairs:
        pair = "".join(c for c in pair if c not in ('!', '.', ':', ',', '"', '?'))
        pair = pair.split()
        w1 = pair[0]
        w2 = pair[1]

        try:
            dataSetI = words.loc[w1].as_matrix()
            dataSetII = words.loc[w2].as_matrix()
            sim_score = spatial.distance.cosine(dataSetI, dataSetII)
            print("{} for '{} {}'".format(round(sim_score,5), w1, w2))
        except KeyError as e:
            not_found_word = str(e).split()[2]
            # print("ERROR: {} not in vocabulary for pair: {} {}".format(not_found_word, w1, w2))
            errors += "{} not in vocabulary for pair: {} {}\n".format(not_found_word, w1, w2)

    print(errors)

def loadGloveModel(glove_file_name, path = "glove_vector_sets/"):
    gloveFile = "{}{}.txt".format(path, glove_file_name)
    print("Loading Glove Model")
    start = timer()
    # https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
    words = pd.read_table(gloveFile, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    stop = timer()
    minutes = (stop-start) // 60
    secs = (stop-start) % 60
    print("Done.",len(words)," words loaded in {} minutes {} seconds".format(minutes, secs))
    words.to_pickle("pickel_files/"+glove_file_name+".pkl")
    return words

if __name__ == "__main__":
    main()