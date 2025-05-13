from train_utils import initiate_tokenizer
import pandas as pd
import json
import lexical_baseline

def read_settings(filepath):

    with open(filepath) as json_file:
        settings = json.load(json_file)

    tokenizer = initiate_tokenizer(settings)
    testfile_name = settings['metadata_testfile']['original_filename']

    return settings, tokenizer, testfile_name

def count_gold_events(tokens, gold):
    """
    Counts amount of tokens annotated with I-event in a given part of the data
    """
    events=0
    zipped = zip(tokens, gold)
    for t, g in zipped:
        tok_lab = zip(t[1:-1], g)
        for tok, lab in tok_lab:
            if lab == 'I-event':
                events+=1
    return events


def calculate_score_event_class(predictions, gold):
    tn=0
    tp=0
    fn=0
    fp=0
    zipped = zip(predictions, gold)
    for p, g in zipped:
        p_g = zip(p, g)
        for pred, gold in p_g:
            if pred == gold and pred == 'O':
                tn+=1
            if pred == gold and pred == 'I-event':
                tp+=1
            if pred != gold and gold == 'I-event':
                fn+=1
            if pred != gold and gold == "O":
                fp+=1

    # allow for no predictions
    try:
        precision = tp / (tp+fp)
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

    return(precision, recall, f1)

def calculate_score_no_class(predictions, gold):
    tn=0
    tp=0
    fn=0
    fp=0
    zipped = zip(predictions, gold)
    for p, g in zipped:
        p_g = zip(p, g)
        for pred, gold in p_g:
            if pred == gold and pred == 'O':
                tp+=1
            if pred == gold and pred == 'I-event':
                tn+=1
            if pred != gold and gold == 'I-event':
                fp+=1
            if pred != gold and gold == "O":
                fn+=1

    #precision = tp / (tp+fp)
    #recall = tp / (tp+fn)
    #f1 = 2 * (precision * recall) / (precision + recall)

    # allow for no predictions
    try:
        precision = tp / (tp+fp)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = tp / (tp+fn)
    except ZeroDivisionError:
        recall = 0

    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    return(precision, recall, f1)

def calculate_macro_avg(precision1, precision2, recall1, recall2):
    mavg_p = (precision1+precision2)/2
    mavg_r = (recall1+recall2)/2
    try:
        mavg_f1 =  2 * ((    ((precision1+precision2)/2)    * ((recall1+recall2)/2)   ) /  (  ((precision1+precision2)/2) + ((recall1+recall2)/2)  ) )
    except ZeroDivisionError:
        mavg_f1 = 0

    return(mavg_p, mavg_r, mavg_f1)

def delete_multiple_at_indices(lst, indices):
    """
    This function is used by token_level_per_sentence, which sends indices of any labels that are not linked to
    any first subtoken of a complete token. This function deletes those labels so that we get matching
    lists of tokens and labels.
    """
    # Sort indices in descending order
    new_list = []
    for i in range(0, len(lst)):
        if i in indices:
            continue
        else:
            new_list.append(lst[i])
    return(new_list)

def token_level_per_sentence(sentence, predictions, gold, tokenizername):   ###Chatgpt was used for this
    """
    According to the tokenizer used, this function joins subtokens back together in the original tokens
    as extracted from Inception and saves the label of the first subtoken of each token.
    This happens on sentence level, i.e., each time one Inception text region is inputted to the function.
    """

    joined_tokens = []
    current_token = ""
    to_delete = []
    if tokenizername.startswith("emanjavacas") and not tokenizername.startswith('emanjavacas/MacBERTh'):
        for i, tok in enumerate(sentence):
            if tok.startswith('##'):
                current_token += tok[2:]  # Append without '##'
                to_delete.append(i)
            else:
                if current_token:  # If there's a current token being built, add it
                    joined_tokens.append(current_token)
                    current_token = ""  # Reset for the next token
                current_token = tok  # Start a new current token with the new token

        # If there's a current token left to add, append it
        if current_token:
            joined_tokens.append(current_token)# + current_token[2:])


    if tokenizername.startswith('emanjavacas/MacBERTh'):
        for i, tok in enumerate(sentence):
            if tok.startswith('##'):
                current_token += tok[2:]  # Append without '##'
                to_delete.append(i)
            else:
                if current_token:  # If there's a current token being built, add it
                    joined_tokens.append(current_token)
                    current_token = ""  # Reset for the next token
                current_token = tok  # Start a new current token with the new token

        # If there's a current token left to add, append it
        if current_token:
            joined_tokens.append(current_token)# + current_token[2:])
            if '[SEP]' in joined_tokens:
                joined_tokens.remove('[SEP]') # filter out [SEP]

    if tokenizername.startswith("GroNLP") or tokenizername.startswith("bert-base-multilingual-cased") or tokenizername.startswith('google-bert'):
        for i, tok in enumerate(sentence):
            if tok.startswith('##'):
                current_token += tok[2:]  # Append without '##'
                to_delete.append(i)
            else:
                if current_token:  # If there's a current token being built, add it
                    joined_tokens.append(current_token)
                    current_token = ""  # Reset for the next token
                current_token = tok  # Start a new current token with the new token

        # If there's a current token left to add, append it
        if current_token:
            joined_tokens.append(current_token)

    new_preds = delete_multiple_at_indices(predictions, to_delete)
    new_gold = delete_multiple_at_indices(gold, to_delete)

    if tokenizername.startswith("FacebookAI/xlm"):
        # Result containers
        joined_tokens = []
        new_preds = []
        new_gold = []

        # Temporary variables for the current word being built
        current_token = ""
        current_label = None
        current_gold = None

        for i, (subtoken, label, gold) in enumerate(zip(sentence, predictions, gold)):
            if subtoken.startswith('▁'):  # new word detected
                # If we already have a word being built, save it
                if current_token:
                    joined_tokens.append(current_token)
                    new_preds.append(current_label)
                    new_gold.append(current_gold)
                # Start a new word
                current_token = subtoken.lstrip('▁')  # Remove leading '▁' for the word
                current_label = label
                current_gold = gold
            else:
                # Append the subtoken to the current word
                current_token += subtoken

            # Handle the last token (flush remaining word)
            if i == len(sentence) - 1:
                joined_tokens.append(current_token)
                new_preds.append(current_label)
                new_gold.append(current_gold)

    if tokenizername.startswith("pdelobelle") or tokenizername.startswith("FacebookAI/roberta") or tokenizername.startswith("glob"):

        # clean up encoding errors
        new_subtokens = []
        for subtoken in sentence:
            if subtoken == 'ĠâĢĶ':
                new_subtokens.append('Ġ—')
            if subtoken == 'âĢĶ':
                new_subtokens.append('—')
            if subtoken == 'ĠÆĴ':
                new_subtokens.append('Ġƒ')
            if subtoken == 'ÆĴ':
                new_subtokens.append('ƒ')
            if subtoken == 'ĠâĢŀ':
                new_subtokens.append('Ġ„')
            if subtoken == 'âĢŀ':
                new_subtokens.append('„')
            if subtoken == 'ĠÂ½':
                new_subtokens.append('Ġ½')
            if subtoken == 'Â½':
                new_subtokens.append('½')
            if subtoken != 'ĠâĢĶ' and subtoken != 'ĠÆĴ' and subtoken != 'ĠâĢŀ' and subtoken != 'ĠÃ½' and subtoken != 'âĢĶ' and subtoken != 'ÆĴ' and subtoken != 'âĢŀ' and subtoken != 'Ã½':
                new_subtokens.append(subtoken)

        joined_tokens = []
        new_preds = []
        new_gold = []

        # Temporary variables for the current word being built
        current_token = ""
        current_label = None
        current_gold = None

        for i, (subtoken, label, gold) in enumerate(zip(new_subtokens, predictions, gold)):
            if subtoken.startswith('Ġ'):  # new word detected
                # If we already have a word being built, save it
                if current_token:
                    joined_tokens.append(current_token)
                    new_preds.append(current_label)
                    new_gold.append(current_gold)
                # Start a new word
                current_token = subtoken.lstrip('Ġ')  # Remove leading '▁' for the word
                current_label = label
                current_gold = gold
            else:
                # Append the subtoken to the current word
                current_token += subtoken

            # Handle the last token (flush remaining word)
            if i == len(sentence) - 1:
                joined_tokens.append(current_token)
                new_preds.append(current_label)
                new_gold.append(current_gold)

    return joined_tokens, new_preds, new_gold

def interpolate(subtokens, predictions, gold, tokenizername):
    interpolated_tokens = []
    interpolated_predictions = []
    interpolated_gold = []
    for t, p, g in zip(subtokens, predictions, gold):
        #print(t, p, g)
        joined_tokens, joined_preds, joined_gold = token_level_per_sentence(t[1:-1], p, g, tokenizername)
        interpolated_tokens.append(joined_tokens)
        interpolated_predictions.append(joined_preds)
        interpolated_gold.append(joined_gold)
    return interpolated_tokens, interpolated_predictions, interpolated_gold


def write_preds_to_file(output_dir, tokens, predictions, gold, settings):
    df = pd.DataFrame() #create dataframe

    #assign columns
    df['token'] = tokens
    df['prediction'] = predictions
    df['gold'] = gold

    lex_tokens = lexical_baseline.parse_lexicon("lexicon_v4.csv") #latest version of the lexicon, also the one released
    lexical_base = lexical_baseline.label_with_lexicon(lex_tokens, tokens)
    df['lexical'] = lexical_base # add lexical baseline predictions to csv with predicted labels per token

    #write to file with filename corresponding to model and test file used
    try:
        df.to_csv(output_dir+'/predictions_'+str(settings['metadata_testfile']['inv_nr'])+'_'+str(settings['metadata_testfile']['year'])+'_'+settings['model'].split('/')[1]+'_'+str(settings['seed'])+'.csv', sep='\t', index=False)
    except IndexError:
        df.to_csv(output_dir + '/predictions_' + str(settings['metadata_testfile']['inv_nr']) + '_' + str(
            settings['metadata_testfile']['year']) + '_' + settings['model'] + '_' + str(
            settings['seed']) + '.csv', sep='\t', index=False)

def get_datastats(tokens, predictions, gold):
    tokens = [item for row in tokens for item in row]
    predictions = [item for row in predictions for item in row]
    gold = [item for row in gold for item in row]

    eventcount_g = 0
    for label in gold:
        if label == 'I-event':
            eventcount_g += 1
    token_count = len(tokens)
    eventcount_p = 0
    for label in predictions:
        if label == 'I-event':
            eventcount_p += 1

    try:
        gold_density = ((eventcount_g / token_count) * 100)
    except ZeroDivisionError:
        gold_density = 0

    return token_count, eventcount_g, eventcount_p, gold_density

def main():
    print("This file should not be run independently ")

if __name__ == "__main__":
    main()



