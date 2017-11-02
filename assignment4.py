import random
import pandas as pd
import csv
from timeit import default_timer as timer

from scipy import spatial

def main():
    f = open("input.txt", "r")
    sentence_list = f.readlines()
    f.close()

    # Remove new line character from end of every sentence
    for i in range(len(sentence_list)):
        sentence_list[i] = sentence_list[i][:-1]
    # print(sentence_list)

    glove_file_name = "glove.840B.300d" # This one takes forever, load it once then use hdf!!!
    # glove_file_name = "glove.6B.50d"
    # loadGloveModel(glove_file_name)
    start = timer()
    words = pd.read_hdf("hdf_files/{}.hdf".format(glove_file_name), 'mydata')
    stop = timer()
    minutes = (stop - start) // 60
    secs = (stop - start) % 60
    print("Read hdf took {} mins {} secs".format(minutes, secs))

    word_pairs = randomlySelectWordPairs(words)

    # not_in_vocab(sentence_list, words)
    calcCosSimilarityWordPairs(words, word_pairs)

    clean_sentence_list = clean_input(sentence_list)
    random_sentence_pairs = randomlySelectSentencePairs(clean_sentence_list)
    calcCosSimilaritySentences(random_sentence_pairs)


def not_in_vocab(sentence_list, words):
    errors = []
    clean_sentence = clean_input(sentence_list)
    for sentence in clean_sentence:
        for word in sentence.split():
            try:
                words.loc[word].as_matrix()
            except KeyError as e:
                not_found_word = str(e).split()[2][1:-1]
                print("{} not in vocabulary\n".format(not_found_word))
                if not not_found_word in errors:
                    errors.append(not_found_word)
    print(errors)


def clean_input(sentence_list):
    # Cleaning so as to reduce words that are not present in vocab
    '''
    Only words in input that are not in the vector .txt (Big one, 840B.300d) (after this cleaning, that is)
    ['BoJack', 'Cordovia']
    '''
    for sentence_index in range(len(sentence_list)):
        sentence = sentence_list[sentence_index]
        sentence = sentence.replace("Mr.", "Mister ")
        sentence = sentence.replace("-", " ")
        sentence = sentence.replace("'re ", " are ")
        sentence = sentence.replace("'ll ", " will ")
        sentence = sentence.replace("'ve ", " have ")
        sentence = sentence.replace("n't ", " not ")
        sentence = sentence.replace("et's ", "et us ") # Fixes [Ll]et's
        sentence = sentence.replace("Horsin' ", "Horsing ")
        sentence = sentence.replace("DE-Nile", "denile")
        sentence = sentence.replace("$500,000", "five hundered thousand dollars")
        sentence = sentence.replace("game's ", "game is ")
        sentence = sentence.replace("everybody's ", "everybody is ")
        sentence = sentence.replace("hat's", "hat is") # that, what
        sentence = sentence.replace("who's ", "who is ")
        sentence = sentence.replace("Time's ", "Time is ")
        sentence = sentence.replace("one's ", "one is ")
        sentence = sentence.replace("ere's ", "ere is ") #there, here
        sentence = sentence.replace("Guy's ", "Guy has ") #there, here

        # As we care more about word meaning than grammatical relationships, we will loose track of possession
        # in this cleaning process
        sentence = sentence.replace("'s", "")

        sentence = "".join(c for c in sentence if c not in ('!', '.', ':', ',', '"', '?', "'"))

        sentence_list[sentence_index] = sentence

    # print("CLEANED", sentence_list)
    return sentence_list

def randomlySelectWordPairs(words, n = 25):
    '''
    Gets list of random word pairs
    :param sentence_list: List of sentences
    :param n: Number word pairs desired (Optional, default = 25)
    :return: List of word pairs
    '''
    word_pairs = []
    for i in range(n):
        w1 = random.choice(words.index)
        w2 = random.choice(words.index)
        word_pair = w1 + " " + w2
        word_pairs.append(word_pair)

    # print(word_pairs)
    return word_pairs

def randomlySelectSentencePairs(sentence_list, n = 25):
    random_sentence_list = []
    for i in range(n):
        s1 = random.choice(sentence_list)
        s2 = random.choice(sentence_list)
        s_pair = [s1, s2]
        random_sentence_list.append(s_pair)

    return random_sentence_list

def calcCosSimilarityWordPairs(words, word_pairs):
    errors = "ERRORS:\n"
    for pair in word_pairs:
        pair = "".join(c for c in pair if c not in ('!', '.', ':', ',', '"', '?'))
        pair = pair.split()
        w1 = pair[0]
        w2 = pair[1]

        try:
            # https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
            dataSetI = words.loc[w1].as_matrix()
            dataSetII = words.loc[w2].as_matrix()
            sim_score = spatial.distance.cosine(dataSetI, dataSetII)
            print("{} for '{} {}'".format(round(sim_score,5), w1, w2))
        except KeyError as e:
            not_found_word = str(e).split()[2]
            # print("ERROR: {} not in vocabulary for pair: {} {}".format(not_found_word, w1, w2))
            errors += "{} not in vocabulary for pair: {} {}\n".format(not_found_word, w1, w2)
    if len(errors) > 9:
        print(errors)

def calcCosSimilaritySentences(random_sentence_pairs):
    for sentence_pair in random_sentence_pairs:
        s1 = sentence_pair[0]
        s2 = sentence_pair[1]
        s1_vector, s2_vector = makeSentenceVectors(s1, s2)
        similarity = spatial.distance.cosine(s1_vector, s2_vector)
        print("{} for '{}.' and '{}.'".format(round(similarity, 5), s1, s2))

def makeSentenceVectors(s1,s2):
    word_list = s1.split() + [i for i in s2.split() if i not in s1]
    s1_vector = [0 for i in word_list]
    s2_vector = [0 for i in word_list]

    for i in range(len(word_list)):
        word = word_list[i]
        if word in s1:
            s1_vector[i] = 1
        if word in s2:
            s2_vector[i] = 1

    return s1_vector, s2_vector

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

    # https://stackoverflow.com/questions/29547522/python-pandas-to-pickle-cannot-pickle-large-dataframes
    words.to_hdf("hdf_files/"+glove_file_name+".hdf", 'mydata', mode='w')
    return words

if __name__ == "__main__":
    main()
