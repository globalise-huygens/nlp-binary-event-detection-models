from transformers import RobertaTokenizerFast, XLMRobertaTokenizerFast, BertTokenizerFast, AutoModelForTokenClassification



#tokenizer = RobertaTokenizerFast.from_pretrained("pdelobelle/robbert-v2-dutch-base")
#tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-multilingual-cased')
#tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
#tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-large-cased')
#tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-large')
#tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-cased')
#tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base')
tokenizer = BertTokenizerFast.from_pretrained('emanjavacas/MacBERTh')

text = 'Den Coromandelsen handel wert volgens uEd: marinen buijten belastingh'
word = 'belastingh'
tokenized_input = tokenizer(word)
tokenized_word = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"]) #check if for all models you should do this
token_id = tokenized_input["input_ids"][1:-1]

print(tokenized_input)
print(tokenized_word)
print()
print()

id2label = {0: 'O', 1: 'I-event'}
label2id = {'O': 0, 'I-event': 1}

model = AutoModelForTokenClassification.from_pretrained(
    'google-bert/bert-base-cased', num_labels=2, id2label=id2label, label2id=label2id
)


print('done')

