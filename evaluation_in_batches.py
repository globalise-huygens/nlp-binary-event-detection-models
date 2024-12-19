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
INV_NRS = ['1160', '1066', '7673', '11012', '9001', '3598', '1348', '1090', '1430', '2665', '1439', '1595', '2693', '3476', '8596','4071']
SEEDS = ['23052024', '21102024', '888', '553311', '6834']


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
    :param seed: str
    :param inv_nr: str
    :param model: str
    """
    # get a list of all filepaths to all documents (all inv_nrs)
    filepaths = get_filepath_list(ROOT_PATH)

    #read in a file: for now hardcoded beginfolder and data of creation, will be updated
    try:
        settings, tokenizer, testfile_names = read_settings("output_in_batches_nov20/"+seed + '_' + model + r"/settings2024-11-20" + "_" + inv_nr + '.json')
    except FileNotFoundError:
        settings, tokenizer, testfile_names = read_settings(
            "output_in_batches/" + seed + '_' + model + r"/settings2024-11-09" + "_" + inv_nr + '.json')

    # read in file for when dealing with globertise
    #settings, tokenizer, testfile_names = read_settings(
       # "output_in_batches_globertise/" + seed + '_' + model + r"/settings2024-12-04" + "_" + inv_nr + '.json')


    prepared_tr, train_data, test_data, prepared_te = construct_datadicts(tokenizer, filepaths, testfile_names)

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

    #uncomment following line if you want to write predictions to df to check and comparepredicted labels per token to gold and to lexical baseline
    #write_preds_to_file('output/checking_output', tokens, predictions, gold, settings)

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
    Creates a 'table' of results per model+seed combination (thus covering all datasplits, i.e. representing 16 fine-tuned models)
    outputs a list of dictionaries, each dictionary being a row in a table. Each dictionary is an output of the 'parse_one_file' function

    :param model: str
    :param seed: str
    """

    all_results = []
    for inv_nr in INV_NRS:
        result_dict = parse_one_file(inv_nr, seed, model)
        all_results.append(result_dict)

    return(all_results)

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
        settings, tokenizer, testfile_names = read_settings(
            "output_in_batches_nov20/" + seed + '_' + model + r"/settings2024-11-20" + "_" + inv_nr + '.json')
        prepared_tr, train_data, test_data, prepared_te = construct_datadicts(tokenizer, filepaths, testfile_names)

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

        accuracy = (overlap_count / len(gold_mentions)) * 100
        accuracies.append(accuracy)


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
    model = 'GysBERT-v2'

    for inv_nr in INV_NRS:
        settings, tokenizer, testfile_names = read_settings(
            "output_in_batches_nov20/" + seed + '_' + model + r"/settings2024-11-20" + "_" + inv_nr + '.json')
        prepared_tr, train_data, test_data, prepared_te = construct_datadicts(tokenizer, filepaths, testfile_names)

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
    outputs a dictionary for one model and its average scores
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

    model = 'GysBERT'
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


def get_average_scores_table():
    modelnames = ['bert-base-dutch-cased', 'GysBERT', 'GysBERT-v2', 'xlm-roberta-base']

    #modelnames = ['globertise']

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

def get_std_dev_per_inv_nr():

    results_per_model = []
    models = ["bert-base-dutch-cased"]
    # table = per model + seed
    for model in models:
        #results_per_model = []
        for seed in SEEDS:
            results = create_table(model, seed)
            results_per_model.append(results)

    f1_dict = {}
    for nr in INV_NRS:
        f1_dict[nr] = []
    for seed_result in results_per_model:
        for dict in seed_result:
            for nr in INV_NRS:
                #f1_invs = []
                if dict['inv_nr'] == nr:
                    #f1_invs.append(dict['f1-event'])
                    f1_dict[dict['inv_nr']].append(dict['R-event'])

    std_devs = []
    for key, value in f1_dict.items():
        sqd_diffs = []
        #print(len(value))
        mean = sum(value)/len(value)
        print(key, mean)
        for score in value:
            sqd_diffs.append((score - mean) ** 2)
          #  print(sqd_diffs)
        std_devs.append(math.sqrt((sum(sqd_diffs) / 5))) # 5 = len(SEEDS)


    return(f1_dict, std_devs)


print("AVERAGE SCORES EVERYTHING")
average_scores = get_average_scores_table()
print(average_scores)

#average_scores_globertise = get_average_scores_table()
#print(average_scores_globertise)

df = pd.DataFrame.from_dict(average_scores)
df.to_csv('AVERAGES.csv', sep='\t')




#f1s, std_devs = get_std_dev_per_inv_nr()
#print(f1s)
#print()
#print(std_devs)

#avg_st_dev = sum(std_devs) / len(std_devs)
#print(avg_st_dev)


#### later, check for all models which documents have lowest mean f1 and highest standard dev

#dev = get_dev_between_seeds("GysBERT", 'f1-event')
#print(dev)