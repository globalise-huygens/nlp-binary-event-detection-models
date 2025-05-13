import ast


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
def tokenize_and_choose_label(tokenizer, corpus):
    """
    Inception outputs tokenized text. This function tokenizes these tokens into subtokens and labels them with their corresponding annotation
    """

    new_corpus = []
    for sen in corpus:
        new_sen = []
        for item in sen:
            word = item[0]
            label = item[1]
            tokenized_input = tokenizer(word)
            tokenized_word = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])[1:-1] #check if for all models you should do this
            token_id = tokenized_input["input_ids"][1:-1]

            if label == 'I-event' or label == 'B-event':
                new_labels = []
                tags = []
                for subword in tokenized_word:
                    new_labels.append(1)
                    tags.append('I-event')
                new_datapoint = (tokenized_word, tags, new_labels, token_id)
                new_sen.append(new_datapoint)
            if label == 'O':
                new_labels = []
                tags = []
                for subword in tokenized_word:
                    new_labels.append(0)
                    tags.append('O')
                new_datapoint = (tokenized_word, tags, new_labels, token_id)
                new_sen.append(new_datapoint)
        new_corpus.append(new_sen)

    return(new_corpus)

def tokenize_and_choose_label_robbert(tokenizer, corpus):
    """
    Inception outputs tokenized text. This function tokenizes these tokens into subtokens and labels them with their corresponding annotation
    especially for RobBERT and RoBERTa: adding a space to each token at the beginning becauase the model otherwise does not record the beginning of a token in their tokenization (missing what is a subtoken and what is a token)
    """
    new_corpus = []
    for sen in corpus:
        new_sen = []
        for item in sen:
            word = item[0]
            label = item[1]
            tokenized_input = tokenizer(' '+word)
            tokenized_word = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])[
                             1:-1]  # check if for all models you should do this
            token_id = tokenized_input["input_ids"][1:-1]

            if label == 'I-event' or label == 'B-event':
                new_labels = []
                tags = []
                for subword in tokenized_word:
                    new_labels.append(1)
                    tags.append('I-event')
                new_datapoint = (tokenized_word, tags, new_labels, token_id)
                new_sen.append(new_datapoint)
            if label == 'O':
                new_labels = []
                tags = []
                for subword in tokenized_word:
                    new_labels.append(0)
                    tags.append('O')
                new_datapoint = (tokenized_word, tags, new_labels, token_id)
                new_sen.append(new_datapoint)
        new_corpus.append(new_sen)

    return (new_corpus)


def prepare_data(corpus):
    """
    Converts the data to input to finetune a LM
    """
    new_tokens = []
    new_labels = []
    tok_per_sen = []
    lab_per_sen = []
    tag_per_sen = []
    input_ids_per_sen = []

    for sentence in corpus:
        temp_tok = []
        temp_lab = []
        temp_inpids = []
        temp_tags = []
        for tup in sentence:
            for token in tup[0]:
                temp_tok.append(token)
                new_tokens.append(token)
            for tag in tup[1]:
                temp_tags.append(tag)
            for label in tup[2]:
                temp_lab.append(label)
                new_labels.append(label)
            for token_id in tup[3]:
                temp_inpids.append(token_id)
        temp_tok.insert(0, '[CLS]')
        temp_tok.append('[SEP]')
        temp_lab.insert(0, -100)
        temp_lab.append(-100)
        temp_inpids.insert(0, 2)
        temp_inpids.append(3)
        temp_tags.insert(0, 'O')
        temp_tags.append('O')
        tag_per_sen.append(temp_tags)
        tok_per_sen.append(temp_tok)
        lab_per_sen.append(temp_lab)
        input_ids_per_sen.append(temp_inpids)

    ### add token type ids and attention mask
    token_type_ids = []
    attention_masks = []
    for sen in tok_per_sen:
        temp_t = []
        temp_a = []
        for i in range(0, len(sen)):
            temp_t.append(0)
            temp_a.append(1)
        token_type_ids.append(temp_t)
        attention_masks.append(temp_a)

    data = {'id': list(range(0, len(tok_per_sen))), 'tokens': tok_per_sen, 'event_tags': tag_per_sen,
            'input_ids': input_ids_per_sen, 'token_type_ids': token_type_ids, 'attention_mask': attention_masks,
            'labels': lab_per_sen}

    return(data)

def restructure_and_truncate(data):

    restructured = []
    for i in range(0, data['id'][-1]+1): #for every paragraph in the datadict
        restructured.append({k: v[i] for k, v in data.items() if k not in ["id"]}) #change structure to list of dicts, 1 dict for evert paragraph,. and exclude information irrelevant for training

    truncated = []
    for par in restructured: #for each paragraph
        assert len(par['input_ids']) == len(par['attention_mask']) == len(par['labels'])
        if len(par['input_ids']) < 512:
            truncated.append(par)
        else:
            for key, value in par.items():
                par[key] = value[:512]
            truncated.append(par)

    return(truncated)

def get_filepaths(filepaths, testfile_names):
    testfile_paths = []
    ### If you selected more than 1 doc to test on
    if type(testfile_names) == list:
        for path in filepaths:
            for file in testfile_names:
                if path.split('/')[-1] == file:
                    testfile_paths.append(path)  # append complete path to chosen files
    else:  # if you just selected one
        for path in filepaths:
            if path.split('/')[-1] == testfile_names:
                testfile_paths.append(path)  # append complete path to chosen file

    trainfile_paths = [x for x in filepaths if x not in testfile_paths] #get all paths that are not testfile paths
    return(trainfile_paths, testfile_paths)

def construct_datadicts(tokenizername, tokenizer, filepaths, testfile_names):
    trainpaths, testpaths = get_filepaths(filepaths, testfile_names)

    corpus_tr = read_data(trainpaths)
    corpus_te = read_data(testpaths)

    if tokenizername.startswith('pdelobelle') or tokenizername.startswith("FacebookAI/roberta") or tokenizername.startswith('glob'): #add spaces to each token when using robbert so that it marks subtokens during procesing
        tokenized_and_labeled_tr = tokenize_and_choose_label_robbert(tokenizer, corpus_tr)
        tokenized_and_labeled_te = tokenize_and_choose_label_robbert(tokenizer, corpus_te)
    else:
        tokenized_and_labeled_tr = tokenize_and_choose_label(tokenizer, corpus_tr)
        tokenized_and_labeled_te = tokenize_and_choose_label(tokenizer, corpus_te)

    prepared_tr = prepare_data(tokenized_and_labeled_tr)
    train_data = restructure_and_truncate(prepared_tr)

    prepared_te = prepare_data(tokenized_and_labeled_te)
    test_data = restructure_and_truncate(prepared_te)

    return(prepared_tr, train_data, test_data, prepared_te)


