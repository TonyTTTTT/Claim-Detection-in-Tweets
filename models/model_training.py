import os
import random
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, AutoConfig,\
    BertForSequenceClassification
from data_loader import DataLoader, compute_metrics
from model_config import model_path, dataset, num_train_epochs, preprocess_function
import transformers
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_preprocess_methods import insert_srl_tag, extract_to_sentence_level, extract_all_frames, none_operation, \
    concate_all_frames, convert_to_srl_tag


dataloader = DataLoader(preprocess_function=preprocess_function, dataset=dataset)
train_dataset, dev_dataset, test_dataset = dataloader.get_dataset(include_test=True)


def set_seed(seed: int = None):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def model_init():
    configuration = AutoConfig.from_pretrained(model_path)
    # configuration.hidden_dropout_prob = 0.3
    # configuration.attention_probs_dropout_prob = 0.1
    model = RobertaForSequenceClassification.from_pretrained(model_path, config=configuration)
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


lr_scheduler_type = "linear"
per_device_train_batch_size = 4
training_args = TrainingArguments(
    output_dir='results',  # model save dir
    logging_dir='./logs/{}/{}_{}_{}_{}'.format(dataset, model_path, dataloader.preprocess_function.__name__, lr_scheduler_type, num_train_epochs),  # directory for storing logs
    evaluation_strategy='epoch',
    # logging_steps=100,
    logging_strategy='epoch',
    # save_steps=10000,
    save_strategy="no",

    per_device_train_batch_size=per_device_train_batch_size,
    # per_device_eval_batch_size=64,

    # learning_rate=3e-5,
    num_train_epochs=num_train_epochs,
    # adam_epsilon=2.5e-9,
    warmup_steps=(len(train_dataset.ids)/per_device_train_batch_size) * 7,
    # weight_decay=0.01,
    # no_cuda=True,
    lr_scheduler_type=lr_scheduler_type,
)

# No1 team treat dev dataset as eval_dataset, so here I do the same
trainer = Trainer(
    model_init=model_init,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)


# set_seed(17)
trainer.train()
# trainer.hyperparameter_search()
print("==========================")
result = trainer.evaluate()
print("==========================")
output = trainer.predict(test_dataset)

if dataloader.preprocess_function == extract_all_frames:
    current_id = None
    predction_sum = None
    predictions = []
    labels = []
    current_id = test_dataset.ids[0]
    predction_sum = np.array(output.predictions[0])
    cnt = 1
    for i in range(1, len(test_dataset.ids)):
        if test_dataset.ids[i] != current_id:
            predictions.append(predction_sum/cnt)
            labels.append(test_dataset.labels[i-1])
            current_id = test_dataset.ids[i]
            cnt = 1
            predction_sum = np.array(output.predictions[i])
        else:
            cnt += 1
            predction_sum += np.array(output.predictions[i])

    predictions.append(predction_sum / cnt)
    labels.append(test_dataset.labels[i])
    predictions = np.array(predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions.argmax(axis=-1), average='binary')
    print('f1 at article-level: {}'.format(f1))

# trainer.save_model('results/final')

# with open('add-srl-longest-tag-64-bertweet-pred-test.tsv', 'w') as f:
#     pred = output[0]
#     pred_argmax = pred.argmax(-1)
#     # score_sum = 0
#     # frame_cnt = 0
#     for idx in range(0, len(pred)):
#         # if idx != 0 and test_dataset.ids[idx] != test_dataset.ids[idx-1]:
#         #     score_avg = score_sum/frame_cnt
#         #     f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score_avg, 'test'))
#         #     score_sum = 0
#         #     frame_cnt = 0
#         score = pred[idx][pred_argmax[idx]]
#         if pred_argmax[idx] == 0:
#             # if the score of label0 is bigger, that it be negative, for create final score
#             score = -1 * score
#         f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score, 'test'))
#         # score_sum += score
#         # frame_cnt += 1
#         # print('{}\t{}\t{}\t{}'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score, 'test'))
#
#     # f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score_avg, 'test'))
