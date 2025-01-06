# nlp-binary-event-detection-models
Code to fine-tune models for binary event detection on annotated documents on an HPC and evaluate locally.

This is work that builds on https://github.com/globalise-huygens/nlp-event-testset-experiment

## Fine-tuning models with 16-fold cross-validation

This repo contains code to fine-tune models on a 16-fold datasplit. For each datasplit, one annotated document is excluded from the training set on which a model is fine-tuned. This document is referred to with its inventory number (as documented in the VOC archives). The excluded document serves as validation data in the evaluation step that happens later. The code in this repo fine-tunes four different models, each with five set seeds. The shell script to run this code is _finetune.sh_. This shell script runs _finetune_with_click.py_, which in turn relies on i) _get_data_selection.py_ for creating the datasplits and storing metadata on each test file, on ii) _train_utils.py_ to initialize the appropriate tokenizer and on iii) _utils.py_ for data (pre)processing. 

The models used in this version of the code are GysBERT, GysBERT-v2, BERTje and XLM-Roberta.

The data on which I fine-tune is stored in "data". _get_data_selection.py_ uses the subfolderstructure as well as the filenames in this directory to make datasplits and gather metadata.

When running _finetune.sh_, 320 seperate models are fine-tuned (4 models x 5 seeds x 16 datasets), of which predictions are seperately stored. _finetune_with_click.py_ exports predictions of each fine-tuned model on the held-out document from the 16 ones available to a json file. In this json file, metadata on the test set and on the training arguments are also stored. Predictions on each of the 16 documents are gathered in folders that represent a model + seed combination (also reflected in the foldername, i.e. "GysBERT-553311"). These folders with predictions can be found in the "output_in_batches_nov20" folder.

## Evaluating the models
__evaluation_in_batches.py__ processes the predictions stored in "output_in_batches_nov20" and the corresponding gold data and evaluates on token level (as opposed to subtoken level) and on mention level. Token level evaluation means binary token classification (IO). See an example underneath.

![Screenshot 2025-01-06 at 10 45 20](https://github.com/user-attachments/assets/de2c8841-82c7-4fcb-b856-fad5fb2caedf)

For token-level binary event detection, precision, recall, and f1 are calculated for the event class (I) as well as for the O class, and a macro-average for each scoretype. Additionally, the same scores are calculated for a lexical baseline. This baseline uses _lexicon_v4_, which was developed in an iterative manner through analysis of annotations, professional expertise of historians working with the VOC archives and a word2vec model trained on the archives. These scores are calculated for each model+seed combination and can be found in the "tables" folder. Each table shows scores for each of the 16 datasplits. It also contains some information about the amount of tokens in each document evaluated on and the event density in the gold data. 

Mention-level scores are calculated as an accuracy score. If one or more of the tokens within a gold event mention span (i.e. "ordonnantie" in "d'ordonnantie ende last") is recognized as an event token in the predictions, we see it as overlap. For the example data given above, the accuracy would be 100%. 

The next step is analysing all results on model-level. In the "results" folder, __AVERAGES.csv__ schows the average score per model (i.e. averaged over seeds and dataplits). 

