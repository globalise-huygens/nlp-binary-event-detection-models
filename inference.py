from train_utils import initiate_tokenizer
import pandas as pd
import json
import lexical_baseline

def read_settings(filepath):

    with open(filepath) as json_file:
        settings = json.load(json_file)

    tokenizer = initiate_tokenizer(settings)
    testfile_names = settings['metadata_testfile']['original_filename']

    return settings, tokenizer, testfile_names

def count_gold_events(tokens, gold):
    events=0
    zipped = zip(tokens, gold)
    for t, g in zipped:
        tok_lab = zip(t[1:-1], g)
        for tok, lab in tok_lab:
            if lab == 'I-event':
                events+=1
    return events


def calculate_score_event_class(predictions, gold):
    """

    """
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

    try:
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        precision = 0
        recall = 0
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

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return(precision, recall, f1)

def calculate_macro_avg(precision1, precision2, recall1, recall2):
    mavg_p = (precision1+precision2)/2
    mavg_r = (recall1+recall2)/2
    mavg_f1 =  2 * ((    ((precision1+precision2)/2)    * ((recall1+recall2)/2)   ) /  (  ((precision1+precision2)/2) + ((recall1+recall2)/2)  ) )

    return(mavg_p, mavg_r, mavg_f1)

def delete_multiple_at_indices(lst, indices):
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

    """

    joined_tokens = []
    current_token = ""
    to_delete = []
    if tokenizername.startswith("emanjavacas"):
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

    if tokenizername.startswith("GroNLP"):
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

    if tokenizername.startswith("pdelobelle"):
        for i, tok in enumerate(sentence):
            #print(tok)
            if tok.startswith('Ġ'):
             #   print('subtoken detected', tok)
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

    if tokenizername.startswith("globert"):
        # Result containers
        joined_tokens = []
        new_preds = []
        new_gold = []

        # read in reformatted vocab
        with open('globertokenizer/tokenizer_from_scratch/reformatted/vocab.json') as infile:
            data = infile.read()
        #vocab_dict = ast.literal_eval(data)

        # Temporary variables for the current word being built
        current_token = ""
        current_label = None
        current_gold = None

        for i, (subtoken, label, gold) in enumerate(zip(sentence, predictions, gold)):
            for i, tok in enumerate(sentence):
                #for key, value in vocab_dict.items():
                if '_'+tok in data:
                    # If we already have a word being built, save it
                    if current_token:
                        joined_tokens.append(current_token)
                        new_preds.append(current_label)
                        new_gold.append(current_gold)
                    # Start a new word
                    current_token = subtoken #.lstrip('▁')  # Remove leading '▁' for the word
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

    if tokenizername.startswith("FacebookAI"):
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

    return joined_tokens, new_preds, new_gold


def interpolate(subtokens, predictions, gold, tokenizername):
    #print(len(subtokens))
    #print(len(predictions))
    #print(len(gold))
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
    #flatten list of lists (list per paragraph)
    tokens = [item for row in tokens for item in row]
    predictions = [item for row in predictions for item in row]
    gold = [item for row in gold for item in row]

    #assign columns
    df['token'] = tokens
    df['prediction'] = predictions
    df['gold'] = gold

    lex_tokens = lexical_baseline.parse_lexicon("lexicon_v3.csv")
    lexical_base = lexical_baseline.label_with_lexicon(lex_tokens, tokens)
    df['lexical'] = lexical_base # add lexical baseline predictions to csv with predicted labels per token

    #write to file with filename corresponding to model and test file used
    df.to_csv(output_dir+'/predictions_'+str(settings['metadata'][0]['inv_nr'])+'_'+str(settings['metadata'][0]['year'])+'_'+settings['model'].split('/')[1]+'.csv', sep='\t', index=False)

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

    gold_density = ((eventcount_g / token_count) * 100)

    return token_count, eventcount_g, eventcount_p, gold_density

def main():
    print("This file should not be run independently ")

if __name__ == "__main__":
    main()



