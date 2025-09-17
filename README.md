# Language Models Lack Temporal Generalization and Bigger is Not Better

This repo contains code and results reported on in our [paper](https://aclanthology.org/2025.findings-acl.1060.pdf)

## General info

Code to fine-tune models for binary event detection on annotated documents on an HPC and evaluate locally.

This is work that builds on https://github.com/globalise-huygens/nlp-event-testset-experiment

## Fine-tuning models with 15-fold cross-validation

This repo contains code to fine-tune models on a 15-fold datasplit. For each datasplit, one annotated document is excluded from the training set on which a model is fine-tuned. This document is referred to with its inventory number (as documented in the VOC archives). The excluded document serves as validation data in the evaluation step that happens later. The code in this repo fine-tunes seven different models, each with five set seeds. The shell script to run this code is _finetune.sh_. This shell script runs _finetune_with_click.py_, which in turn relies on i) _get_data_selection.py_ for creating the datasplits and storing metadata on each test file, on ii) _train_utils.py_ to initialize the appropriate tokenizer and on iii) _utils.py_ for data (pre)processing. 

The models used in this version of the code are 5 Dutch models and 2 multilingual ones, namely GysBERT, GysBERT-v2, BERTje, RobBERT, XLM-Roberta and Multilingual BERT, and our own model, GloBERTise (see our [other repo](https://github.com/globalise-huygens/GloBERTise) for code and documentation on how we created it - the model itself can be downloaded [here](https://huggingface.co/globalise/GloBERTise)). Changing some variables in the code enables you to use some English models; BERT, RoBERTa, MacBERTh.

The data on which I fine-tune is stored in "data". _get_data_selection.py_ uses the subfolderstructure as well as the filenames in this directory to make datasplits and gather metadata.

When running _finetune.sh_, 450 seperate models are fine-tuned (4 models x 5 seeds x 15 datasets), of which predictions are seperately stored. _finetune_with_click.py_ exports predictions of each fine-tuned model on the held-out document from the 15 ones available to a json file. In this json file, metadata on the test set and on the training arguments are also stored. Predictions on each of the 15 documents are gathered in folders that represent a model + seed combination (also reflected in the foldername, i.e. "GysBERT-553311"). These folders with predictions can be found in the "output_in_batches_*" folders.

## Evaluating the models
__evaluation_in_batches.py__ processes the predictions stored in "output_in_batches_*" and the corresponding gold data and evaluates on token level (as opposed to subtoken level) and on mention level. Token level evaluation means binary token classification (IO). See an example underneath.

![Screenshot 2025-01-06 at 10 45 20](https://github.com/user-attachments/assets/de2c8841-82c7-4fcb-b856-fad5fb2caedf)

For token-level binary event detection, precision, recall, and f1 are calculated for the event class (I) as well as for the O class, and a macro-average for each scoretype. Additionally, the same scores are calculated for a lexical baseline. This baseline uses _lexicon_v4_, which was developed in an iterative manner through analysis of annotations, professional expertise of historians working with the VOC archives and a word2vec model trained on the archives. This lexicon only matches single tokens to an event class. These scores are calculated for each model+seed combination and can be found in the "tables" folder. Each table shows scores for each of the 15 datasplits. It also contains some information about the amount of tokens in each document evaluated on and the event density in the gold data. 

Mention-level scores are calculated as an accuracy score. If one or more of the tokens within a gold event mention span (i.e. "ordonnantie" in "d'ordonnantie ende last") is recognized as an event token in the predictions, we see it as overlap. For the example data given above, the accuracy would be 100%. 

The next step is analysing all results on model-level. In the "results" folder, __AVERAGES.csv__ schows the average score per model (i.e. averaged over seeds and datasplits). For example, the scores for xlm-r are the averages of the scores reorted in tables "table_xlm-roberta-base_6834.csv", "table_xlm-roberta-base_888.csv", "table_xlm-roberta-base_553311.csv", "table_xlm-roberta-base_21102024.csv", and "table_xlm-roberta-base_23052024.csv". In the averages table, only precision, recall and f1 for the event class are reported, as this is the only score that is truly of interest. 

## CRF baseline
I train a Conditional Random Forest with features to compare the Language Models' performance with (as well as a lexical baseline). The main feature for this model are token embeddings derived with a specialized word2vec model, trained on our VOC corpus. The model and its documentation can be found [here](https://zenodo.org/records/15038313). To run crf_baselines.py you will need to download the word2vec model (and save it locally, in the code the hardcoded filepath is 'word2vec/GLOBALISE.word2vec')

## Lexical baseline
Our lexical baseline is thoroughly documented [here](https://github.com/globalise-huygens/nlp-event-lexical-approach). I also wrote a [blogpost](https://globalise.huygens.knaw.nl/yes-okay-but-what-were-they-doing/) on it.

