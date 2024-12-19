from transformers import RobertaTokenizer, BertTokenizerFast, AutoTokenizer, RobertaTokenizerFast, XLMRobertaTokenizerFast

def initiate_tokenizer(settings):
    if settings['tokenizer'] == 'emanjavacas/GysBERT-v2':
        tokenizer = BertTokenizerFast.from_pretrained('emanjavacas/GysBERT-v2')#, model_max_length = 512) # works without model_max_lenght, makes no difference in performance
    if settings['tokenizer'] == 'emanjavacas/GysBERT':
        tokenizer = BertTokenizerFast.from_pretrained('emanjavacas/GysBERT')#, padding=True)
    if settings['tokenizer'] == 'FacebookAI/xlm-roberta-base':
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
    if settings['tokenizer'] == 'pdelobelle/robbert-v2-dutch-base':
        tokenizer = RobertaTokenizerFast.from_pretrained('pdelobelle/robbert-v2-dutch-base')
    if settings['tokenizer'] == 'GroNLP/bert-base-dutch-cased':
        tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    if settings['tokenizer'] == 'globertise':
        #tokenizer = AutoTokenizer.from_pretrained('../globertise/tokenizers/tokenizer_from_scratch')
        tokenizer = AutoTokenizer.from_pretrained('globertokenizer/tokenizer_from_scratch')
    return(tokenizer)


