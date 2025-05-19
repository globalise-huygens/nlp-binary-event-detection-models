### initial code from https://www.geeksforgeeks.org/conditional-random-fields-crfs-for-pos-tagging-in-nlp/
import sklearn_crfsuite
from sklearn_crfsuite import metrics as metrics
import numpy as np
import ast
from gensim.models import Word2Vec
from utils import get_filepaths
from get_data_selection import get_filepath_list, file_selection_invnr
import pandas as pd
from evaluation_in_batches import find_event_groups

ROOT_PATH = 'data/json_per_doc'
W2VK = Word2Vec().wv.load_word2vec_format('word2vec/GLOBALISE.word2vec')


def read_data(paths):
    """
    Reads a jsonfile and returns the corpus as a list of list with a sentence er list
    """

    corpus = []
    for path in paths:
        with open(path, 'r') as file:
            data = file.read()

        list_data = []
        for item in data.split('\n'):
            list_data.append(ast.literal_eval(item))

        for d in list_data:
            sentence = []
            for i in range(len(d['words'])):
                sentence.append((d['words'][i], d['events'][i]))
            corpus.append(sentence)

    return(corpus)

def construct_datasplit(testfile_name):
    filepaths = get_filepath_list(ROOT_PATH)
    trainpaths, testpaths = get_filepaths(filepaths, testfile_name)

    corpus_tr = read_data(trainpaths)
    corpus_te = read_data(testpaths)

    return(corpus_tr, corpus_te)

def get_vector_features(word, model):
    word=word.lower()
    try:
         vector=model[word]
    except:
        # if the word is not in vocabulary,
        # returns zeros array
        vector=np.zeros(300,)

    return vector


def get_word_features(sentence, i, w2vmodel, setting):
    word = sentence[i][0]

    if setting == 'base':
        features = {
            'word': word,
        }

    if setting == 'base-plus':
        features = {
            'word': word,
             #extracting previous word
            'prev_word': '' if i == 0 else sentence[i-1][0],
             #extracting next word
            'next_word': '' if i == len(sentence)-1 else sentence[i+1][0]
        }

    if setting == 'base-extra':
        features = {
            'word': word,
            'is_first': i == 0, #if the word is a first word
            'is_last': i == len(sentence) - 1,  #if the word is a last word
            'is_capitalized': word[0].upper() == word[0],
            'is_all_caps': word.upper() == word,      #word is in uppercase
            'is_all_lower': word.lower() == word,      #word is in lowercase
            # prefix of the word
            'prefix-1': word[0],
            'prefix-2': word[:2],
            'prefix-3': word[:3],
            # suffix of the word
            'suffix-1': word[-1],
            'suffix-2': word[-2:],
            'suffix-3': word[-3:],
            # extracting previous word
            'prev_word': '' if i == 0 else sentence[i - 1][0],
            # extracting next word
            'next_word': '' if i == len(sentence) - 1 else sentence[i + 1][0],
            'has_hyphen': '-' in word,    #if word has hypen
            'is_numeric': word.isdigit(),  #if word is in numeric
            'capitals_inside': word[1:].lower() != word[1:]
        }

    if setting == 'embedding-extra':
        wordembedding = get_vector_features(word, w2vmodel)
        features = {
            'word': word,
            #'embedding': [embedding],
            'is_first': i == 0,  # if the word is a first word
            'is_last': i == len(sentence) - 1,  # if the word is a last word
            'is_capitalized': word[0].upper() == word[0],
            'is_all_caps': word.upper() == word,  # word is in uppercase
            'is_all_lower': word.lower() == word,  # word is in lowercase
            # prefix of the word
            'prefix-1': word[0],
            'prefix-2': word[:2],
            'prefix-3': word[:3],
            # suffix of the word
            'suffix-1': word[-1],
            'suffix-2': word[-2:],
            'suffix-3': word[-3:],
            # extracting previous word
            'prev_word': '' if i == 0 else sentence[i - 1][0],
            # extracting next word
            'next_word': '' if i == len(sentence) - 1 else sentence[i + 1][0],
            'has_hyphen': '-' in word,  # if word has hypen
            'is_numeric': word.isdigit(),  # if word is in numeric
            'capitals_inside': word[1:].lower() != word[1:]
        }

        for iv, value in enumerate(wordembedding):
            features['v{}'.format(iv)] = value

    if setting == 'embedding-only':
        wordembedding = get_vector_features(word, w2vmodel)
        features = {
            'word': word
        }
        for iv, value in enumerate(wordembedding):
            features['v{}'.format(iv)] = value


    return(features)

def convert_to_IO(labels):
    """
    Converts BIO-labelling to IO-labelling
    """

    new_labels = [[x if x != 'B-event' else 'I-event' for x in sl] for sl in labels]

    return(new_labels)

def extract_features_per_sentence(corpus, w2vmodel, setting):
    X = []
    y = []
    for sentence in corpus:
        X_sentence = []
        y_sentence = []
        for i in range(len(sentence)):
            X_sentence.append(get_word_features(sentence, i, w2vmodel, setting))
            try:
                y_sentence.append(sentence[i][1])
            except IndexError:
                y_sentence.append('X')
        X.append(X_sentence)
        y.append(y_sentence)

    return(X, y)

def get_train_test(corpus_tr, corpus_te,  W2VK, setting, problem_type = 'IO'):
    X_train, y_train = extract_features_per_sentence(corpus_tr, W2VK, setting)
    X_test, y_test =  extract_features_per_sentence(corpus_te, W2VK, setting)

    if problem_type == 'IO':
        y_test = convert_to_IO(y_test)
        y_train = convert_to_IO(y_train)
        X_train = convert_to_IO(X_train)
        X_test = convert_to_IO(X_test)

    return(X_train, y_train, X_test, y_test)

def define_simple_crf():
    """
    fits a crf, returns predictions on X_test
    """
    # Train a CRF model on the training data
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    return(crf)

def fit_and_predict(X_train, y_train, X_test, crf):

    crf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = crf.predict(X_test)

    return(y_pred)

def print_scores(y_test, y_pred):

    print()
    print('accuracy: ', metrics.flat_accuracy_score(y_test, y_pred))
    print('macro precision: ', metrics.flat_precision_score(y_test, y_pred, average = 'macro'))
    print('macro recall: ', metrics.flat_recall_score(y_test, y_pred, average = 'macro'))
    print('macro f1: ', metrics.flat_f1_score(y_test, y_pred, average = 'macro'))


def calculate_accuracy(y_test, y_pred):
    """
      Calculates the mean accuracy for mention span overlap over all inventory numbers of one model + seed combination
      :param seed: str
      :param inv_nr: str
      :param model: str
      """

    predicted = [item for row in y_pred for item in row]
    gold = [item for row in y_test for item in row]

    gold_mentions = find_event_groups(gold)
    predicted_mentions = find_event_groups(predicted)

    overlap_count = 0
    for gm in gold_mentions:
        for pm in predicted_mentions:
            # Check if there's an overlap by seeing if there's any intersection between the two groups
            if any(i in gm for i in pm):
                overlap_count += 1
                break  # Once we find an overlap for this group, no need to check further for this group


    accuracy = (overlap_count / len(gold_mentions)) * 100

    return (accuracy)

def calculate_score_event_class(y_test, y_pred):

    predictions = []
    for sen in y_pred:
        for label in sen:
            predictions.append(label)

    gold = []
    for sen in y_test:
        for label in sen:
            gold.append(label)

    tn = 0
    tp = 0
    fn = 0
    fp = 0
    zipped = zip(predictions, gold)
    for pred, gold in zipped:
        if pred == gold and pred == 'O':
            tn += 1
        if pred == gold and pred == 'I-event':
            tp += 1
        if pred != gold and gold == 'I-event':
            fn += 1
        if pred != gold and gold == "O":
            fp += 1

    # allow for no predictions
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    accuracy = calculate_accuracy(y_test, y_pred)

    return (precision, recall, f1, accuracy)

def train_and_evaluate_per_testfile(testfilename):
    corpus_tr, corpus_te = construct_datasplit(testfilename)
    X_train, y_train, X_test, y_test = get_train_test(corpus_tr, corpus_te, W2VK, 'embedding-extra', problem_type = 'IO')

    crf = define_simple_crf()
    y_pred = fit_and_predict(X_train, y_train, X_test, crf)


    print('macro scores:')
    print_scores(y_test, y_pred)
    print()
    print('scores for event class:')
    precision, recall, f1, accuracy = calculate_score_event_class(y_test, y_pred)
    print('event precision: ', precision)
    print('event recall: ', recall)
    print('event f1: ', f1)

    return(precision, recall, f1, accuracy)

def get_average_scores():

    all_precision = []
    all_recall = []
    all_f1 = []
    all_accuracy = []

    inv_nrs = ['1160', '1066', '7673', '11012', '9001', '1348', '4071', '1090', '1430', '2665', '1439', '1595', '2693',
               '3476', '8596']
    for inv_nr in inv_nrs:
        settings = file_selection_invnr(ROOT_PATH, inv_nr)
        precision, recall, f1, accuracy = train_and_evaluate_per_testfile(settings['metadata_testfile']['original_filename'])
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_accuracy.append(accuracy)

    avg_precision = sum(all_precision) / len(inv_nrs)
    avg_recall = sum(all_recall) / len(inv_nrs)
    avg_f1 = sum(all_f1) / len(inv_nrs)
    avg_accuracy = sum(all_accuracy) / len(all_accuracy)

    average_scores = {}
    average_scores['CRF'] = {'P-event': avg_precision, 'R-event': avg_recall, 'f1-event:': avg_f1, 'mean_accuracy': avg_accuracy}

    print(average_scores)

    return(average_scores)


average_scores = get_average_scores()
df = pd.DataFrame.from_dict(average_scores)
df.to_csv('results/Averages_precision_recall_f1/AVERAGES-CRF-new.csv', sep='\t')


print("end")