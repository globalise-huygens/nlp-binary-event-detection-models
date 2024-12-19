import ast
import torch
import transformers
from transformers import set_seed, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from train_utils import initiate_tokenizer
from get_data_selection import get_filepath_list, file_selection_invnr
from utils import construct_datadicts
from datasets import Dataset
import evaluate
import numpy
import json
import os
from datetime import date
import click

def create_settings(root_path, inv_nr, tokenizername, modelname, seed, label_list):
    settings = file_selection_invnr(root_path, inv_nr)
    settings['tokenizer'] = tokenizername
    settings['model'] = modelname
    settings['seed'] = seed
    settings['label_list'] = ast.literal_eval(label_list)
    return(settings)

def initiate(settings, root_path):
    tokenizer = initiate_tokenizer(settings)
    testfile_names = settings['metadata_testfile']['original_filename']
    filepaths = get_filepath_list(root_path)

    id2label = {0: 'O', 1: 'I-event'}
    label2id = {'O': 0, 'I-event': 1}

    model = AutoModelForTokenClassification.from_pretrained(
        settings['model'], num_labels=2, id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    return(tokenizer, testfile_names, filepaths, model, data_collator)



@click.command()
@click.option('--seed', type=click.INT)
@click.option('--inv_nr', type=click.STRING)
@click.option('--root_path', type=click.STRING)
@click.option('--tokenizername', type=click.STRING)
@click.option('--modelname', type=click.STRING)
@click.option('--label_list', type=click.STRING)
def main(root_path, inv_nr, tokenizername, modelname, seed, label_list):

    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    today = date.today()
    seqeval = evaluate.load("seqeval")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    settings = create_settings(root_path, inv_nr, tokenizername, modelname, seed, label_list)
    tokenizer, testfile_names, filepaths, model, data_collator = initiate(settings, root_path)
    prepared_tr, train_data, test_data, prepared_te = construct_datadicts(tokenizer, filepaths, testfile_names)

    train = Dataset.from_list(train_data)
    test = Dataset.from_list(test_data)

    for param in model.parameters(): param.data = param.data.contiguous()

    print("VERSIONS")
    print(transformers.__version__)
    print(evaluate.__version__)

    # on snellius: 4.32.1
    # 0.4.2

    def compute_metrics(p):
        """
        computes scores per eval_step and saves predictions
        """
        predictions, labels = p
        predictions = numpy.argmax(predictions, axis=2)

        print(predictions)

        true_predictions = [
            [settings['label_list'][p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        settings['predictions'] = true_predictions

        true_labels = [
            [settings['label_list'][l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        settings['gold'] = true_labels

        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        settings['results'] = {
            "precision": results["overall_precision"],  # subtoken level
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    learning_rate = 5e-5
    per_device_train_batch_size = 32
    per_device_test_batch_size = 32
    num_train_epochs = 5
    weight_decay = 0.01

    settings['training_args'] = {'learning_rate': learning_rate, 'per_device_train_batch_size': per_device_train_batch_size, 'num_train_epochs': num_train_epochs, 'weight_decay': weight_decay}

    training_args = TrainingArguments(
        output_dir="models",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_test_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch", # eval_strategy when using transformers 4.43.2
        save_strategy="no",
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("Start training")

    trainer.train()
    modelname = settings['model'].split('/')[1]

    with open('output_in_batches_globertise/'+str(settings['seed'])+'_'+modelname+'/settings'+str(today)+'_'+str(settings['metadata_testfile']['inv_nr'])+'.json', 'w') as fp:
        json.dump(settings, fp)

if __name__ == '__main__':
    main()