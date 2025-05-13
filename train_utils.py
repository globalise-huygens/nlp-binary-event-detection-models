from transformers import BertTokenizerFast, AutoTokenizer, RobertaTokenizerFast, XLMRobertaTokenizerFast

def initiate_tokenizer(settings):
    if settings['tokenizer'] == 'emanjavacas/GysBERT-v2':
        tokenizer = BertTokenizerFast.from_pretrained('emanjavacas/GysBERT-v2')#, model_max_length = 512) # works without model_max_lenght, makes no difference in performance
    if settings['tokenizer'] == 'emanjavacas/GysBERT':
        tokenizer = BertTokenizerFast.from_pretrained('emanjavacas/GysBERT')#, padding=True)
    if settings['tokenizer'] == 'FacebookAI/xlm-roberta-base':
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
    if settings['tokenizer'] == 'pdelobelle/robbert-v2-dutch-base':
        tokenizer = RobertaTokenizerFast.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    if settings['tokenizer'] == 'GroNLP/bert-base-dutch-cased':
        tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    if settings['tokenizer'] == 'bert-base-multilingual-cased':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    if settings['tokenizer'] == 'google-bert/bert-base-cased':
        tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-cased')
    if settings['tokenizer'] == 'FacebookAI/roberta-base':
        tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base')
    if settings['tokenizer'] == 'FacebookAI/roberta-large':
        tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-large')
    if settings['tokenizer'] == 'google-bert/bert-large-cased':
        tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-large-cased')
    if settings['tokenizer'] == 'emanjavacas/MacBERTh':
        tokenizer = BertTokenizerFast.from_pretrained('emanjavacas/MacBERTh')
    if settings['tokenizer'] == 'globalise/GloBERTise':
        tokenizer = RobertaTokenizerFast.from_pretrained('globalise/GloBERTise')
    return(tokenizer)


