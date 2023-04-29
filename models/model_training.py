import os
import random
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, AutoConfig,\
    BertForSequenceClassification
from data_loader import DataLoader, compute_metrics
from model_config import *
import transformers
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data_preprocess_methods import split_into_sentences, split_into_frames

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(device)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits").to(device)
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.73, 1.0])).to(device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


dataloader = DataLoader(preprocess_function=preprocess_function, dataset=dataset, do_normalize=do_normalize, concate_frames_num=concate_frames_num)
train_dataset, dev_dataset, test_dataset = dataloader.get_dataset(include_test=True)


def model_init():
    configuration = AutoConfig.from_pretrained(model_path)
    # configuration.hidden_dropout_prob = 0.3
    # configuration.attention_probs_dropout_prob = 0.1

    if model_path.startswith('roberta'):
        model = RobertaForSequenceClassification.from_pretrained(model_path, config=configuration)
    elif model_path.startswith('bert'):
        model = BertForSequenceClassification.from_pretrained(model_path, config=configuration)

    # model.classifier.dropout.p = 0.5
    # for i in range(len(trainer.model.roberta.encoder.layer)):
    #     try:
    #         print(trainer.model.roberta.encoder.layer[i].droupout)
    #     except:
    #         print('{} layer no dropout'.format(i))
    # model.roberta.encoder.layer.__delitem__(9)
    # model.roberta.encoder.layer.__delitem__(8)
    # model.roberta.encoder.layer.__delitem__(7)
    # model.roberta.encoder.layer.__delitem__(6)

    # for param in model.roberta.parameters():
    #     param.requires_grad = False

    return model


training_args = TrainingArguments(
    output_dir='results',  # model save dir
    logging_dir='./logs/{}/{}_{}_{}_{}'.format(dataset, model_path, dataloader.preprocess_function.__name__, lr_scheduler_type, num_train_epochs),  # directory for storing logs
    evaluation_strategy='epoch',
    # logging_steps=100,
    logging_strategy='epoch',
    # save_steps=10000,
    save_strategy="no",
    logging_first_step=True,

    per_device_train_batch_size=per_device_train_batch_size,
    # per_device_eval_batch_size=64,

    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    # adam_epsilon=2.5e-9,
    warmup_steps=(len(train_dataset.ids)/(per_device_train_batch_size * device_num)) * warm_up_epochs,
    # weight_decay=0,
    # no_cuda=True,
    lr_scheduler_type=lr_scheduler_type,
    # seed=42,
)

# No1 team treat dev dataset as eval_dataset, so here I do the same
trainer = Trainer(
    model_init=model_init,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)


trainer.train()
# trainer.hyperparameter_search()
print("==========================")
result = trainer.evaluate()
print("==========================")
output = trainer.predict(test_dataset)
print("==========================")


def calculate_article_score_from_sentence(test_dataset, output, combine_method):
    current_id = None
    prediction_sum = None
    predictions = []
    labels = []
    current_id = test_dataset.ids[0]
    prediction_sum = np.array(output.predictions[0])
    prediction_max = np.array(output.predictions[0])
    cnt = 1
    for i in range(1, len(test_dataset.ids)):
        if test_dataset.ids[i] != current_id:
            if combine_method == 'avg':
                predictions.append(prediction_sum/cnt)
            elif combine_method == 'max':
                predictions.append(prediction_max)
            labels.append(test_dataset.labels[i-1])
            current_id = test_dataset.ids[i]
            cnt = 1
            prediction_sum = np.array(output.predictions[i])
            prediction_max = np.array(output.predictions[i])
        else:
            cnt += 1
            prediction_sum += np.array(output.predictions[i])
            if output.predictions[i][1] - output.predictions[i][0] > prediction_max[1] - prediction_max[0]:
                prediction_max = np.array(output.predictions[i])

    if combine_method == 'avg':
        predictions.append(prediction_sum / cnt)
    elif combine_method == 'max':
        predictions.append(prediction_max)
    labels.append(test_dataset.labels[i])
    predictions = np.array(predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions.argmax(axis=-1), average='binary')
    acc = accuracy_score(labels, predictions.argmax(axis=-1))
    confusionMatrix = confusion_matrix(labels, predictions.argmax(axis=-1))
    print('\n=================================\n'
          'using method: {}'
          'f1 at article-level: {}\nacc at article-level: {}\nconfusion matrix: {}'
          '\n=================================\n'.format(combine_method, f1, acc, confusionMatrix))


preprocess_dataset_name = dataloader.preprocess_dataset_name

dataloader = DataLoader(preprocess_function=split_into_sentences, dataset=preprocess_dataset_name,
                        do_normalize=do_normalize, concate_frames_num=concate_frames_num)
train_dataset, dev_dataset, test_dataset = dataloader.get_dataset(include_test=True)
output = trainer.predict(test_dataset)
calculate_article_score_from_sentence(test_dataset, output, 'max')

dataloader = DataLoader(preprocess_function=split_into_frames, dataset=preprocess_dataset_name,
                        do_normalize=do_normalize, concate_frames_num=concate_frames_num)
train_dataset, dev_dataset, test_dataset = dataloader.get_dataset(include_test=True)
output = trainer.predict(test_dataset)
calculate_article_score_from_sentence(test_dataset, output, 'max')


# trainer.save_model('results/final')

# write result to clef evaluation format
# with open('none-operation-64-bertweet-test.tsv', 'w') as f:
#     pred = output[0]
#     pred_argmax = pred.argmax(-1)
#     f.write('topic\ttweet_id\ttweet_url\ttweet_text\tclass_label\n')
#     # score_sum = 0
#     # frame_cnt = 0
#     for idx in range(0, len(pred)):
#         # if idx != 0 and test_dataset.ids[idx] != test_dataset.ids[idx-1]:
#         #     score_avg = score_sum/frame_cnt
#         #     f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score_avg, 'test'))
#         #     score_sum = 0
#         #     frame_cnt = 0
#
#         # score = pred[idx][pred_argmax[idx]]
#         # if pred_argmax[idx] == 0:
#         #     # if the score of label0 is bigger, that it be negative, for create final score
#         #     score = -1 * score
#
#         score = pred[idx].argmax()
#         f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score, 'test'))
#         # score_sum += score
#         # frame_cnt += 1
#         # print('{}\t{}\t{}\t{}'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score, 'test'))
#
#     # f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score_avg, 'test'))
