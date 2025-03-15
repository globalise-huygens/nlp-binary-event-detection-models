### from evaluation_in_batches.py

def preprocess_robbert(predictions, subtokens, example_subtokens, gold):
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
            print("long subtokens: ", new_sen)
            new_sen.append('[SEP]')
            truncated_subtokens.append(new_sen)

    truncated_example_subtokens = []
    for sen in example_subtokens:
        if len(sen) < 512:
            truncated_example_subtokens.append(sen)
        else:
            new_sen = sen[:512]
            print("long example subtokens: ", new_sen)
            new_sen.append('[SEP]')
            truncated_example_subtokens.append(new_sen)


    print('running preprocess_robbert')
    interpolated_tokens, interpolated_predictions, interpolated_gold = interpolate_robbert(truncated_example_subtokens, truncated_subtokens, predictions,
                                                                                   gold)

    return(interpolated_tokens, interpolated_predictions, interpolated_gold)


#from parse_one_file

  if model != 'robbert-v2-dutch-base':
        interpolated_tokens, interpolated_predictions, interpolated_gold = preprocess_tokens(predictions, subtokens, gold, settings)

    if model == 'robbert-v2-dutch-base':
        interim_tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
        example_prepared_te = construct_datadicts(interim_tokenizer, filepaths, testfile_names)[3]
        example_subtokens = example_prepared_te['tokens'].copy()
        interpolated_tokens, interpolated_predictions, interpolated_gold = preprocess_robbert(predictions, subtokens, example_subtokens, gold)



#########from inference.py
def token_level_per_sentence_robbert(example_subtokens, subtokens, predictions, gold):
    #Get example of token representations using xlm-r

    joined_tokens = []

    # Temporary variables for the current word being built
    current_token = ""

    for i, subtoken in enumerate(example_subtokens):
        if subtoken.startswith('▁'):  # new word detected
            if current_token:    # If we already have a word being built, save it
                joined_tokens.append(current_token)
            # Start a new word
            current_token = subtoken.lstrip('▁')  # Remove leading '▁' for the word
        else:
            # Append the subtoken to the current word
            current_token += subtoken

        # Handle the last token (flush remaining word)
        if i == len(example_subtokens) - 1:
            joined_tokens.append(current_token)


    #print('joined tokens: ', joined_tokens)

    new_subtokens = []
    for subtoken in subtokens:
        if subtoken == 'âĢĶ':
           # print('subtoken: ', subtoken)
            new_subtokens.append('—')
        if subtoken == 'ÆĴ':
            #print('subtoken: ', subtoken)
            new_subtokens.append('ƒ')
        if subtoken == 'âĢŀ':
            new_subtokens.append('„')
        if subtoken == 'Â½':
            new_subtokens.append('½')
        if subtoken != 'âĢĶ' and subtoken != 'ÆĴ' and subtoken != 'âĢŀ' and subtoken != 'Ã½':
            new_subtokens.append(subtoken)

    #print(new_subtokens)



    result_labels = []
    result_gold = []

    # Initialize index to iterate over the subtokens and labels
    subtoken_idx = 0

    # Loop through token representations
    for token in joined_tokens:
        # Find how many subtokens this token corresponds to
        current_token = ''

        # Keep adding subtokens until the token matches the full representation
        first_subtoken_label = None
        first_subtoken_gold = None

        while len(current_token) < len(token):
            current_token += new_subtokens[subtoken_idx]

            # Store the label of the first subtoken of the current token
            if first_subtoken_label is None:
                first_subtoken_label = predictions[subtoken_idx]

            if first_subtoken_gold is None:
                first_subtoken_gold = gold[subtoken_idx]

            subtoken_idx += 1

        result_labels.append(first_subtoken_label)
        result_gold.append(first_subtoken_gold)

        if subtoken_idx == len(new_subtokens):  # check if we've reached the last subtoken
            break



    # Output the result
    #print(result_labels)



    #result_labels = []
    #result_gold = []

    # Index to keep track of where we are in the subtokens list
    #subtoken_index = 0



    # Loop through each token
    #for token in tokens:
        #print(token)
        # Determine the start and end index of subtokens that correspond to this token
        #token_length = len(token)  # Get length of the token
        #token_subtokens = []


        # Add subtokens for this token

        #while subtoken_index < len(new_subtokens) and sum(len(sub) for sub in token_subtokens) + len(
               # new_subtokens[subtoken_index]) <= token_length:
            #token_subtokens.append(new_subtokens[subtoken_index])
            #subtoken_index += 1

        #print(token_subtokens)
        # The label for the token will be the label of the first subtoken
        #first_subtoken_index = new_subtokens.index(token_subtokens[0])
       # print(first_subtoken_index)
        #token_label = predictions[first_subtoken_index]
        #gold_label = gold[first_subtoken_index]

        # Append the label for the current token to the result list
        #result_labels.append(token_label)
        #result_gold.append(gold_label)

    # Output result
    print(joined_tokens)
    print(len(joined_tokens))
    print(result_labels)
    print(len(result_labels))
    print(result_gold)
    print(len(result_gold))

    if len(joined_tokens) > len(result_labels):
        joined_tokens = joined_tokens[:len(result_labels)]

    return(joined_tokens, result_labels, result_gold)

def interpolate_robbert(example_st, subtokens, predictions, gold):
    interpolated_tokens = []
    interpolated_predictions = []
    interpolated_gold = []
    for e_st, st, p, g in zip(example_st, subtokens, predictions, gold):
        print('e_st: ', e_st)
        print('st: ', st)
        #print(p)
        #print(g)
        # print(t, p, g)
        joined_tokens, joined_preds, joined_gold = token_level_per_sentence_robbert(e_st[1:-1], st[1:-1], p, g)
        interpolated_tokens.append(joined_tokens)
        interpolated_predictions.append(joined_preds)
        interpolated_gold.append(joined_gold)
    return interpolated_tokens, interpolated_predictions, interpolated_gold