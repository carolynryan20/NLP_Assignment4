from collections import OrderedDict
from csv import QUOTE_NONE
from timeit import default_timer as timer
from numpy import dot, linalg
from errno import ENOENT
from random import choice as random_choice
from pandas import read_table as pd_read_table
from pandas import read_hdf as pd_read_hdf
from os import makedirs, strerror
from os.path import isfile, isdir
from sys import path as sys_path
from sys import argv as sys_argv

# NLP Assignment 4
# Carolyn Ryan

def main():
    '''
    Run part one and two
    '''
    if len(sys_argv) > 1:
        glove_file_name = sys_argv[1]
    else:
        glove_file_name = "glove.6B.50d"
        print("Attempting to use glove.6B.50d for vector configuration, if you want a different vector file, please use command line to input one\n"
              "Form: python3 assignment4.py glove.6B.50d")
    words = getModel(glove_file_name)

    # PART ONE
    word_pairs = randomlySelectWordPairs(words.index.values)
    calcCosSimilarityWordPairs(words, word_pairs)

    # PART TWO
    with open("input.txt", "r") as f:
        sentence_list = f.readlines()

    # Remove new line character from end of every sentence
    sentence_list = [sentence[:-1] for sentence in sentence_list]

    random_sentence_pairs = randomlySelectSentencePairs(sentence_list)
    calcCosSimilaritySentences(random_sentence_pairs)

def randomlySelectWordPairs(word_list, n = 25):
    '''
    Gets list of random word pairs
    :param word_list: List of words in vector set : list of str
    :param n: Number word pairs desired (Optional, default = 25) : int
    :return: List of word pairs : list of str
    '''
    word_pairs = []
    for i in range(n):
        w1 = random_choice(word_list)
        w2 = random_choice(word_list)
        word_pair = w1 + " " + w2
        word_pairs.append(word_pair)

    return word_pairs

def randomlySelectSentencePairs(sentence_list, n = 25):
    '''
    Given a list of sentences, randomly selects pairs of sentences
    :param sentence_list: A list of sentences: list of strings
    :param n: Number sentence pairs desired, default = 25 : int
    :return: List of randomly paired sentences: list of lists of str
    '''
    random_sentence_list = []
    for i in range(n):
        s1 = random_choice(sentence_list)
        s2 = random_choice(sentence_list)
        s_pair = [s1, s2]
        random_sentence_list.append(s_pair)

    return random_sentence_list

def calcCosSimilarityWordPairs(words, word_pairs):
    '''
    Calculate the cosine similarity scores given a list of word pairs
    https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    :param words: word and vector representations : dataframe
    :param word_pairs: Word pairs list : list of strings
    :return: None (Output printing of word similarity)
    '''
    print("Cosine Similarity for Random Word Pairs")
    sim_score_dict = {}
    for pair in word_pairs:
        pair_split = pair.split()
        w1 = pair_split[0]
        w2 = pair_split[1]

        v1 = words.loc[w1].as_matrix()
        v2 = words.loc[w2].as_matrix()
        sim_score = cosine_similarity(v1, v2)

        if sim_score in sim_score_dict:
            sim_score_dict[sim_score].append(pair)
        else:
            sim_score_dict[sim_score] = [pair]
    sim_score_dict = OrderedDict(sorted(sim_score_dict.items()))
    for key,value in sim_score_dict.items():
        print("{} for '{}'".format(key, value))

def calcCosSimilaritySentences(random_sentence_pairs):
    '''
    Calculate cosine similarity scores for sentences
    :param random_sentence_pairs: List of list of sentences
    :return: None (Output is printed similarity scores of sentences)
    '''
    print("\nCosine Similarity for Sentence Pairs")
    sim_score_dict = {}
    for sentence_pair in random_sentence_pairs:
        s1 = sentence_pair[0]
        s2 = sentence_pair[1]
        s1_vector, s2_vector = makeSentenceVectors(s1, s2)
        similarity = cosine_similarity(s1_vector, s2_vector)

        if similarity in sim_score_dict:
            sim_score_dict[similarity].append(sentence_pair)
        else:
            sim_score_dict[similarity] = [sentence_pair]

    sim_score_dict = OrderedDict(sorted(sim_score_dict.items()))
    for key, value in sim_score_dict.items():
        for item in value:
            print("{} for '{}'".format(key, item))

def makeSentenceVectors(s1,s2):
    '''
    Make sentence vectors given two sentences
    Vectors are length of union of all words
    And that word at that index = 1 if the word it corresponds to is in the given sentence
    :param s1: Sentence 1: str
    :param s2: Sentence 2: str
    :return: sentence_one_vector, sentence_one_vector: list of 0/1
    '''
    s1 = s1.split()
    s2 = s2.split()

    word_list = make_word_list(s1, s2)

    s1_vector = [0 for i in word_list]
    s2_vector = [0 for i in word_list]

    for i in range(len(word_list)):
        word = word_list[i]
        if word in s1:
            s1_vector[i] = 1
        if word in s2:
            s2_vector[i] = 1

    return s1_vector, s2_vector


def make_word_list(s1, s2):
    '''
    A method that makes the word list
    And in a hack-y way ensures that repeat words are dealt with as we would want them to be
    :param s1: List of all words in s1
    :param s2: List of all words in s2
    :return: List of all words in both, including repeats when appropriate
    '''
    for i in range(len(s1)):
        word = s1[i]
        for j in range(i + 1, len(s1)):
            if s1[j] == word:
                s1[j] = word + "1"
    for i in range(len(s2)):
        word = s2[i]
        for j in range(i + 1, len(s2)):
            if s2[j] == word:
                s2[j] = word + "1"
    word_list = s1 + [i for i in s2 if i not in s1]
    return word_list


def cosine_similarity(v, w):
    '''
    A function to calculate cosine similarity of two vectors
    Similarity = (v•w)/(|v||w|)
    :param v: First vector : list of numbers
    :param w: Second vector : list of numbers
    :return: Cosine similarity score : float
    '''
    dot_product = dot(v,w)
    mag_v = linalg.norm(v)
    mag_w = linalg.norm(w)

    cosine_similar = dot_product / (mag_v * mag_w)
    return float(cosine_similar)

def getModel(glove_file_name, hdf_path = "hdf_files/", txt_path = "glove_vector_sets/"):
    '''
    Checks if/where glove file could be, if hdf version exists, we want to use that as it will be quickest,
    otherwise we can use the .txt file, if neither exists, raise a file not found error
    :param glove_file_name: Name of glove vector files we are using : str
    :param hdf_path: Path to glove hdf files : str
    :param txt_path: Path to glove vector sets (text files) : str
    :return: word and corresponding vectors : dataframe
    '''
    PATH = sys_path[0] + "/"
    hdf_path_to_file = PATH + hdf_path + glove_file_name + ".hdf"
    txt_path_to_file = PATH + txt_path + glove_file_name + ".txt"

    if isfile(hdf_path_to_file):
        return readGloveModel(glove_file_name, hdf_path)
    elif isfile(txt_path_to_file):
        words = loadGloveModel(glove_file_name, txt_path)
        if not isdir(PATH + hdf_path):
            makedirs(PATH + hdf_path)
        exportGloveModel(words, hdf_path_to_file)
        return words
    else:
        print("Desired glove model not accessible in hdf or txt format.\n"
              "This could be a result of looking in the wrong directory.\n"
              "Raising File Not Found Error")
        raise FileNotFoundError(ENOENT, strerror(ENOENT), txt_path_to_file)


def loadGloveModel(glove_file_name, path):
    '''
    Load glove model into dataframe from .txt file
    https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
    :param glove_file_name: Name of glove vector files we are using : str
    :param path: Path to glove text files : str
    :return: Words dataframe of word and vector matches
    '''
    gloveFile = "{}{}.txt".format(path, glove_file_name)
    print("Loading Glove Model, will take a while")
    start = timer()
    words = pd_read_table(gloveFile, sep=" ", index_col=0, header=None, quoting=QUOTE_NONE)
    stop = timer()
    minutes = (stop-start) // 60
    secs = (stop-start) % 60
    print("Done.",len(words),"words loaded in {} minutes {} seconds\n".format(minutes, secs))
    return words

def exportGloveModel(words, hdf_file_path):
    '''
    Export glove model to hdf for later use
    https://stackoverflow.com/questions/29547522/python-pandas-to-pickle-cannot-pickle-large-dataframes
    :param words: Words vector dataframe to export: dataframe
    :param hdf_file_path: Path to export to : str
    :return: None (saved hdf of glove vector)
    '''
    print("Exporting word vector to hdf to increase runtime of subsequent runs")
    start = timer()
    words.to_hdf(hdf_file_path, 'mydata', mode='w')
    stop = timer()
    print("Done exporting hdf in {} seconds.  HDF now in hdf_files/ for future use".format(round(stop - start,5)))

def readGloveModel(glove_file_name, path):
    '''
    Read Model from hdf
    :param glove_file_name: Name of glove vector files we are using: str
    :param path: Path to hdfFiles: str
    :return: Words dataframe of word and vector matches
    '''
    glove_file = "{}{}.hdf".format(path, glove_file_name)
    print("Reading Glove Model hdf")
    start = timer()
    words = pd_read_hdf(glove_file, 'mydata')
    stop = timer()
    minutes = (stop - start) // 60
    secs = (stop - start) % 60
    print("Read hdf took {} mins {} secs\n".format(minutes, secs))
    return words

if __name__ == "__main__":
    main()
