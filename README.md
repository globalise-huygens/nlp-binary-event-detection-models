# nlp-binary-event-detection-models
Code to fine-tune models for binary event detection on annotated documents on an HPC and evaluate locally.

This is work that builds on https://github.com/globalise-huygens/nlp-event-testset-experiment

## Fine-tuning models with 16-fold cross-validation

This repo contains code to fine-tune models on a 16-fold datasplit. For each datasplit, one annotated document is excluded from the training set on which a model is fine-tuned. This document is referred to with its inventory number (as documented in the VOC archives). The excluded document serves as validation data in the evaluation step that happens later. The code in this repo fine-tunes four different models, each with five set seeds. The shell script to run this code is _finetune.sh_. This shell script runs _finetune_with_click.py_, which in turn relies on i) _get_data_selection.py_ for creating the datasplits and storing metadata on each test file, on ii) _train_utils.py_ to initialize the appropriate tokenizer and on iii) _utils.py_ for data (pre)processing. 

The models used in this version of the code are GysBERT, GysBERT-v2, BERTje and XLM-Roberta.

The data on which I fine-tune is stored in "data". _get_data_selection.py_ uses the subfolderstructure as well as the filenames in this directory to make datasplits and gather metadata.

When running _finetune.sh_, 320 seperate models are fine-tuned (4 models x 5 seeds x 16 datasets), of which predictions are seperately stored. _finetune_with_click.py_ exports predictions of each fine-tuned model on the held-out document from the 16 ones available to a json file. In this json file, metadata on the test set and on the training arguments are also stored. Predictions on each of the 16 documents are gathered in folders that represent a model + seed combination (also reflected in the foldername, i.e. "GysBERT-553311"). These folders with predictions can be found in the "output_in_batches_nov20" folder.

## Evaluating the models



