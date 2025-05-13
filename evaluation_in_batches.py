from get_data_selection import get_filepath_list
from inference import read_settings
from inference import calculate_score_event_class, calculate_score_no_class, calculate_macro_avg, count_gold_events
from inference import interpolate, get_datastats, write_preds_to_file
from utils import construct_datadicts
import lexical_baseline
from sklearn import metrics as mtr
import pandas as pd
import math


ROOT_PATH = "data/json_per_doc/"
INV_NRS = ['1160', '1066', '7673', '11012', '9001', '1090', '1430', '2665', '1439', '1595', '2693', '3476', '8596', '1348', '4071'] #, '3598']
SEEDS = ['21102024', '23052024', '888', '553311', '6834']
INPUT_FOLDER = 'output_in_batches_GloBERTise/'
DATE = '2025-05-13' #'2025-02-04' ' #'2025-04-24' #'2025-04-11'  #'2025-02-04' #for english models '2025-02-10' #for Dutch models '2025-02-04' # date on which models were fine-tuned and predictions were written to settingsfile, i.e. date on which 'finetune.sh' was run.


def preprocess_tokens(predictions, subtokens, gold, settings):
    """
    truncates sequences of subtokens to max 512 and recreates tokens from subtokens

    :param predictions: list
    :param subtokens: list
    :param gold: list
    """

    truncated_subtokens = []
    for sen in subtokens:
        if len(sen) < 512:
            truncated_subtokens.append(sen)
        else:
            new_sen = sen[:512]
            new_sen.append('[SEP]')
            truncated_subtokens.append(new_sen)

    interpolated_tokens, interpolated_predictions, interpolated_gold = interpolate(truncated_subtokens, predictions,
                                                                                   gold, settings['tokenizer'])

    return(interpolated_tokens, interpolated_predictions, interpolated_gold)


def parse_one_file(inv_nr, seed, model, create_confusion_matrix=False):
    """
    This function reads one file of predictions (outputted by finetune_with_click.py).
    These predictions are from one finetuned model on one specific datasplit with one specific seed.
    It prepares the test data in the same way the model processed it while predicting on it and prepares it to be analyzed on token level.
    This involves tokenizing, truncating and mapping subtoken representations back to token representations, saving the prediction of the first subtoken to represent the complete token.
    The function proceeds to calculate scores and outputs a dict containing various scores and metadata.
    The outputted dictionary also contains scores of the lexical baseline.

    :param seed: str
    :param inv_nr: str
    :param model: str

    :param create_confusion_matrix: str --> set to True to show a plotted confusion matrix for the test file + predictions analyzed
    """

    # get a list of all filepaths to all documents (all inv_nrs)
    filepaths = get_filepath_list(ROOT_PATH)

    # read settings that contain predictions and metadata, load tokenizer and get name of test doc
    settings, tokenizer, testfile_name = read_settings(
        INPUT_FOLDER + seed + '_' + model + r"/settings"+DATE+ "_" + inv_nr + '.json')

    # save name of the tokenizer
    tokenizername = settings['tokenizer']

    # prepare data of test documents in the same way we prepared train documents for finetuning
    prepared_tr, train_data, test_data, prepared_te = construct_datadicts(tokenizername, tokenizer, filepaths, testfile_name)

    predictions = settings['predictions'].copy()
    subtokens = prepared_te['tokens'].copy()
    gold = settings['gold']

    interpolated_tokens, interpolated_predictions, interpolated_gold = preprocess_tokens(predictions, subtokens, gold, settings)

    #Scores for class I-event
    p_ev, r_ev, f1_ev = calculate_score_event_class(interpolated_predictions, interpolated_gold)

    #Scores for class "O"
    p_no, r_no, f1_no = calculate_score_no_class(interpolated_predictions, interpolated_gold)

    #Macro scores on token level
    mavg_p, mavg_r, mavg_f1 = calculate_macro_avg(p_ev, p_no, r_ev, r_no)

    #Flatten data accross text regions
    tokens = [item for row in interpolated_tokens for item in row]
    predictions = [item for row in interpolated_predictions for item in row]
    gold = [item for row in interpolated_gold for item in row]

    #uncomment following line if you want to write predictions to df to check and compare predicted labels per token to gold and to lexical baseline
   # write_preds_to_file('checking_output/GloBERTise', tokens, predictions, gold, settings)

    #Create confusion matrix per file
    if create_confusion_matrix == True:
        confusion_matrix = mtr.confusion_matrix(gold, predictions, labels=['O', 'I-event'])
        cm_display = mtr.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        cm_display.plot()

    #Get scores of lexical baseline as well as csv with baseline predictions
    lex_tokens = lexical_baseline.parse_lexicon("lexicon_v4.csv")
    lexical_base = lexical_baseline.label_with_lexicon(lex_tokens, tokens)
    lex_precision, lex_recall, lex_f1 = lexical_baseline.evaluate_baseline(lexical_base, gold)

    token_count, eventcount_g, eventcount_p, gold_density = get_datastats(interpolated_tokens, interpolated_predictions, interpolated_gold)

    assert inv_nr == str(settings['metadata_testfile']['inv_nr'])
    result_dict = {'inv_nr':inv_nr, 'year':settings['metadata_testfile']['year'],'round':settings['metadata_testfile']['round'], '#tokens': token_count, '#gold_event_tokens': eventcount_g, '#predicted_event_tokens': eventcount_p, 'gold_event_density':gold_density, 'P-event':p_ev, 'R-event':r_ev, 'f1-event':f1_ev, 'lex_precision': lex_precision, 'lex_recall': lex_recall, 'lex_f1': lex_f1, 'P-none':p_no, 'R-none':r_no, 'f1-none':f1_no, 'P-macro':mavg_p, 'R-macro':mavg_r, 'f1-macro':mavg_f1}

    return result_dict
def create_table(model, seed):
    """
    Creates a 'table' of results per model+seed combination (thus covering all datasplits, i.e. representing 15 fine-tuned models)
    outputs a list of dictionaries, each dictionary being a row in a table. Each dictionary is an output of the 'parse_one_file' function

    :param model: str
    :param seed: str
    """

    all_results = []
    for inv_nr in INV_NRS:
        result_dict = parse_one_file(inv_nr, seed, model)
        all_results.append(result_dict)

    df = pd.DataFrame.from_dict(all_results)
    df.to_csv('tables/table_'+model+'_'+seed+'.csv')

    return(all_results)


def find_event_groups(series):
    """
    This function is used by the code calculating event mention span detection accuracy
    Looks for subsequent tokens labeled with I-event in a list of gold or predicted labels and stores indexes of groups in list of list
    ChatGPT was used for the creation of this function

    :param series: list
    """
    groups = []
    group = []
    for i, label in enumerate(series):
        if label == 'I-event':
            group.append(i)
        else:
            if group:
                groups.append(group)
                group = []
    if group:  # Append the last group if it's an 'I' group
        groups.append(group)
    return groups


def calculate_mean_accuracy(model, seed):
    """
    Calculates the mean accuracy for mention span overlap over all inventory numbers of one model + seed combination
    :param seed: str
    :param inv_nr: str
    :param model: str
    """
    filepaths = get_filepath_list(ROOT_PATH)

    accuracies = []

    for inv_nr in INV_NRS:
        # read settings that contain predictions and metadata, load tokenizer and get name of test doc
        settings, tokenizer, testfile_name = read_settings(
            INPUT_FOLDER + seed + '_' + model + r"/settings" + DATE + "_" + inv_nr + '.json')

        tokenizername = settings['tokenizer']
        prepared_tr, train_data, test_data, prepared_te = construct_datadicts(tokenizername, tokenizer, filepaths, testfile_name)

        predictions = settings['predictions'].copy()
        subtokens = prepared_te['tokens'].copy()
        gold = settings['gold']

        interpolated_tokens, interpolated_predictions, interpolated_gold = preprocess_tokens(predictions, subtokens,
                                                                                             gold, settings)
        # Create csv with predictions mapped against gold per file
        tokens = [item for row in interpolated_tokens for item in row]
        predictions = [item for row in interpolated_predictions for item in row]
        gold = [item for row in interpolated_gold for item in row]

        ####CALCULATE MENTION SPAN OVERLAP ---> chatgpt was used for this
        gold_mentions = find_event_groups(gold)
        predicted_mentions = find_event_groups(predictions)

        overlap_count = 0
        for gm in gold_mentions:
            for pm in predicted_mentions:
                # Check if there's an overlap by seeing if there's any intersection between the two groups
                if any(i in gm for i in pm):
                    overlap_count += 1
                    break  # Once we find an overlap for this group, no need to check further for this group

        try:
            accuracy = (overlap_count / len(gold_mentions)) * 100
            accuracies.append(accuracy)
        except ZeroDivisionError:
            accuracies.append(0)


    mean_accuracy =  sum(accuracies) / len(accuracies)

    return(mean_accuracy)

def calculate_mean_lexical_accuracy(seed):
    """
      Calculates the mean accuracy for mention span overlap over all inventory numbers of one model + seed combination
      :param seed: str
      :param inv_nr: str
      :param model: str
      """
    filepaths = get_filepath_list(ROOT_PATH)

    accuracies = []
    model = 'GloBERTise' #this defines which tokenizer you use, depending on the tokenizer you use, lexical baseline results will slightly differ #in paper: mBERT

    for inv_nr in INV_NRS:

        # read settings that contain predictions and metadata, load tokenizer and get name of test doc
        settings, tokenizer, testfile_name = read_settings(
            INPUT_FOLDER + seed + '_' + model + r"/settings" + DATE + "_" + inv_nr + '.json')

        tokenizername=settings['tokenizer']
        prepared_tr, train_data, test_data, prepared_te = construct_datadicts(tokenizername, tokenizer, filepaths, testfile_name)

        predictions = settings['predictions'].copy()
        subtokens = prepared_te['tokens'].copy()
        gold = settings['gold']

        truncated_subtokens = []
        for sen in subtokens:
            if len(sen) < 512:
                truncated_subtokens.append(sen)
            else:
                new_sen = sen[:512]
                new_sen.append('[SEP]')
                truncated_subtokens.append(new_sen)

        interpolated_tokens, interpolated_predictions, interpolated_gold = interpolate(truncated_subtokens, predictions,
                                                                                       gold, settings['tokenizer'])

        tokens = [item for row in interpolated_tokens for item in row]
        gold = [item for row in interpolated_gold for item in row]


        ####CALCULATE MENTION SPAN OVERLAP ---> chatgpt was used for this
        lex_tokens = lexical_baseline.parse_lexicon("lexicon_v4.csv")
        lexical_base = lexical_baseline.label_with_lexicon(lex_tokens, tokens)

        gold_mentions = find_event_groups(gold)
        predicted_mentions = find_event_groups(lexical_base)

        overlap_count = 0
        for gm in gold_mentions:
            for pm in predicted_mentions:
                # Check if there's an overlap by seeing if there's any intersection between the two groups
                if any(i in gm for i in pm):
                    overlap_count += 1
                    break  # Once we find an overlap for this group, no need to check further for this group


        accuracy = (overlap_count / len(gold_mentions)) * 100
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)

    return (mean_accuracy)

def calculate_std_dev(results_all_seeds, scoretype):

    avg_scores = []
    for seed in results_all_seeds:
        scores_per_seed= []
        for d in seed:
            scores_per_seed.append(d[scoretype])
        avg_scores.append(sum(scores_per_seed)/15)


    mean = sum(avg_scores)/5

    sqd_diffs = []
    for score in avg_scores:
        sqd_diffs.append((score-mean)**2)

    std_dev = math.sqrt((sum(sqd_diffs)/5))

    return mean, std_dev

def get_dev_between_seeds(model, scoretype): # works but is veryyy slow
    """
    creates all tables for one model and different seeds and calculates deviation in f1 scores for the event class
    :param scoretype" str, indicates which score you want to calculate standard dev for, i.e. "f1-event", "P-event", "R-event"
    """
    all_results = []
    for seed in SEEDS:
        results_per_seed = create_table(model, seed)
        all_results.append(results_per_seed)

    mean, std_dev = calculate_std_dev(all_results, scoretype)

    return std_dev


def get_mean_accuracy_over_seeds(model):

    accuracies = []
    for seed in SEEDS:
        int_mean = calculate_mean_accuracy(model, seed) # mean over inventory numbers but not over seeds
        accuracies.append(int_mean)
    mean = sum(accuracies) / len(accuracies) # mean over inventory numbers and seeds
    return(mean)

def get_mean_lexical_accuracy_over_seeds():

    accuracies = []
    for seed in SEEDS:
        int_mean = calculate_mean_lexical_accuracy(seed) # mean over inventory numbers but not over seeds
        accuracies.append(int_mean)
    mean = sum(accuracies) / len(accuracies) # mean over inventory numbers and seeds
    return(mean)


def get_average_scores(model):
    """
    outputs a dictionary for one model (i.e. XLM-R) and its average scores
    """

    scoretypes = ['P-event', 'R-event', 'f1-event']
    scores = {}

    for score in scoretypes:
        scores_seeds = []
        for seed in SEEDS:
            table = create_table(model, seed)
            scores_inv_nrs = []
            for d in table:
                scores_inv_nrs.append(d[score])
            scores_seeds.append(sum(scores_inv_nrs)/len(INV_NRS))
        scores[score] = sum(scores_seeds)/len(SEEDS)
    return(scores)

def get_average_lexical_scores():
    scoretypes = ['lex_precision', 'lex_recall', 'lex_f1']
    scores = {}

    model = 'GloBERTise' #this defines which tokenizer you use, depending on the tokenizer you use, lexical baseline results will slightly differ #results in paper: mBERT
    for score in scoretypes:
        scores_seeds = []
        for seed in SEEDS:
            table = create_table(model, seed)
            scores_inv_nrs = []
            for d in table:
                scores_inv_nrs.append(d[score])
            scores_seeds.append(sum(scores_inv_nrs)/len(INV_NRS))
        scores[score] = sum(scores_seeds)/len(SEEDS)

    converted_dict = {}
    for key, value in scores.items():
        if key == 'lex_precision':
            converted_dict['P-event'] = value
        if key == 'lex_recall':
            converted_dict['R-event'] = value
        if key == 'lex_f1':
            converted_dict['f1-event'] = value

    return(converted_dict)


def get_average_scores_table(modelnames):
    """
    Creates a dictionary with average scores for several models
    """

    all_scores = {}
    for model in modelnames:
        score_dict = get_average_scores(model)
        mean_accuracy = get_mean_accuracy_over_seeds(model)
        score_dict['mean_accuracy'] = mean_accuracy
        all_scores[model] = score_dict

    lexical_score_dict = get_average_lexical_scores()
    lexical_mean_accuracy = get_mean_lexical_accuracy_over_seeds()
    lexical_score_dict['mean_accuracy'] = lexical_mean_accuracy
    all_scores['lexical_baseline'] = lexical_score_dict

    return(all_scores)

def get_std_dev_per_inv_nr(model, scoretype):

    results_per_model = []

    for seed in SEEDS:
        results = create_table(model, seed)
        results_per_model.append(results)

    score_dict = {}
    for nr in INV_NRS:
        score_dict[nr] = []
    for seed_result in results_per_model:
        for dict in seed_result:
            for nr in INV_NRS:
                if dict['inv_nr'] == nr:
                    score_dict[dict['inv_nr']].append(dict[scoretype])

    std_devs = []
    for key, value in score_dict.items():
        sqd_diffs = []
        mean = sum(value)/len(value)
        for score in value:
            sqd_diffs.append((score - mean) ** 2)
        std_devs.append(math.sqrt((sum(sqd_diffs) / 5))) # 5 = len(SEEDS)


    return(score_dict, std_devs)

def get_avg_std_comp1():
    """
    Calculates standard deviation betweens seeds by  calculating standard deviation between the average scores of a model+seed combination
    """

    models = ["GysBERT", "GysBERT-v2", "xlm-roberta-base", "bert-base-dutch-cased", 'robbert-v2-dutch-base', 'bert-base-multilingual-cased']
    scoretypes = ['P-event', 'R-event', 'f1-event']
    scores = {}

    for model in models:
        per_model = {}
        for scoretype in scoretypes:
            std_dev = get_dev_between_seeds(model, scoretype)
            per_model[scoretype] = std_dev
            scores[model] = per_model

    return(scores)

def get_avg_std_comp2(models):
    """
    Calculates standard deviation between seeds by calculating standard deviation between models fine-tuned on the same datasplit and with the same seed and averaging standard deviation scores over datasplits and seeds per model afterwards
    """

    scoretypes = ['P-event', 'R-event', 'f1-event']
    scores = {}

    for model in models:
        per_model = {}
        for scoretype in scoretypes:
            score_dict, std_devs = get_std_dev_per_inv_nr(model, scoretype)
            per_model[scoretype] = sum(std_devs) / len(std_devs)
            scores[model] = per_model

    return(scores)


def get_f1(model):

    """
    outputs a dictionary of f1 scores for each model+seed combination per modeltype
    """

    scoretypes = ['f1-event']
    scores = {}

    for score in scoretypes:
        scores_seeds = {}
        for seed in SEEDS:
            table = create_table(model, seed)
            scores_inv_nrs = []
            for d in table:
                scores_inv_nrs.append(d[score])
            scores_seeds[seed] = ((sum(scores_inv_nrs) / len(INV_NRS))*1000000)
        return (scores_seeds)

def get_all_f1():

    """
    gets f1 scores for all models
    outputs a dictionary with models as keys and dictionaries as values, this dictionary has seeds as keys and f1 scores as values
    """

    models = ["GloBERTise", "GysBERT", "GysBERT-v2", "xlm-roberta-base", "bert-base-dutch-cased", 'robbert-v2-dutch-base', 'bert-base-multilingual-cased']

    complete_dict = {}
    for model in models:
        scoredict = get_f1(model)
        complete_dict[model] = scoredict

    return(complete_dict)


def get_best_model():
    """
    shows for each seed which model scores the highest f1 score averaged over datasplits as well as which model scores the lowest. Prints the seed, the highest/lowest scoring model and the f1 score
    outputs a csv to the results folder with average f1 scores for each model+seed combination (averaged over datasplits)
    """

    all_f1 = get_all_f1()
    df = pd.DataFrame.from_dict(all_f1)
    df.to_csv("results/Results_per_seed/f1_per_seed_.csv")

    #print worst model per seed
    print("Worst model per seed")
    for seed in SEEDS:
        print(seed, max(int(d[seed]) for d in all_f1.values()), max(all_f1.keys(), key=lambda x: all_f1[x][seed]))

    #print best model per seed
    print("Best model per seed")
    for seed in SEEDS:
        print(seed, min(int(d[seed]) for d in all_f1.values()), min(all_f1.keys(), key=lambda x: all_f1[x][seed]))


def scores_per_inv(model, inv_nr):
    """
    Gets f1 scores with metadata per inventory number, i.e. per datasplit, averaged over seeds

    :param model: str
    :param inv_nr: str
    """

    info = {}
    info['inv_nr'] = inv_nr
    scores_per_seed = []
    pred_events_per_seed = []
    for seed in SEEDS:
        result_dict = parse_one_file(inv_nr, seed, model)
        scores_per_seed.append(round(result_dict['f1-event'], 2))
        pred_events_per_seed.append(result_dict['#predicted_event_tokens'])
        info['year'] = result_dict['year']
        info['#tokens'] = result_dict['#tokens']
        info['#gold_event_tokens'] = result_dict['#gold_event_tokens']
        info['#predicted_event_tokens'] = sum(pred_events_per_seed) / 5
        info['gold_event_density'] = round(result_dict['gold_event_density'], 2)
        info['all_f1'] = scores_per_seed
        info['average_f1'] = round(sum(scores_per_seed) / 5, 2)

    return(info)

def create_table_for_datasplits(model):

    """
    Creates a 'table' of results per model+inventory number combination (averaging over seeds)
    outputs a list of dictionaries, each dictionary being a row in a table. Each dictionary is an output of the 'parse_one_file' function
    """

    all_results = []
    for inv_nr in INV_NRS:
        result_dict = scores_per_inv(model, inv_nr)
        all_results.append(result_dict)

    return(all_results)


def write_datasplit_table_per_model():
    models = ["GloBERTise", "GysBERT", "GysBERT-v2", "xlm-roberta-base", "bert-base-dutch-cased", 'robbert-v2-dutch-base', 'bert-base-multilingual-cased']

    for model in models:
        model_per_fold = create_table_for_datasplits(model)
        df = pd.DataFrame.from_dict(model_per_fold)
        df.to_csv('results/Results_per_datasplit/'+model+'.csv')


def get_main_results():
    """
    Runs code to
    - get average percision, recall, f1 scores per model (i.e., XLM-R) and
    - get standard deviation scores per model

    Outputs csv files for both results
    """

    #defining models to be evaluated
    modelnames = ['GloBERTise']
    #modelnames = ['bert-base-multilingual-cased', "bert-base-dutch-cased","GysBERT", "GysBERT-v2", "xlm-roberta-base", 'robbert-v2-dutch-base']
    #modelnames = ["MacBERTh", "bert-base-cased", 'roberta-base']


    print("AVERAGE SCORES")
    average_scores = get_average_scores_table(modelnames)
    print(average_scores)

    #write scores to file
    df = pd.DataFrame.from_dict(average_scores)
    df.to_csv('results/Averages_precision_recall_f1/AVERAGES-GloBERTise.csv', sep='\t')
    print()

    print("Standard Deviation")
    scores_std_2=get_avg_std_comp2(modelnames)
    print()
    #print(scores_std_2)
    df = pd.DataFrame.from_dict(scores_std_2)
    df.to_csv('results/Standard_deviation/stand_dev_comp2-GloBERTise.csv', sep='\t')

get_main_results()
