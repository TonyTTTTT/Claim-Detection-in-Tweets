import torch.utils.data.dataset
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from data_preprocess import get_dataset, compute_metrics


# model = RobertaForSequenceClassification.from_pretrained("vinai/bertweet-covid19-base-uncased")
train_dataset, dev_dataset = get_dataset()


def model_init():
    configuration = AutoConfig.from_pretrained("vinai/bertweet-covid19-base-uncased")
    configuration.hidden_dropout_prob = 0.3
    configuration.attention_probs_dropout_prob = 0.3
    model = RobertaForSequenceClassification.from_pretrained("vinai/bertweet-covid19-base-uncased", config=configuration)
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
    return model


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    logging_dir='./logs',  # directory for storing logs
    evaluation_strategy='epoch',
    logging_steps=100,
    save_steps=100,
    per_device_train_batch_size=16,  # batch size per device during training
    # per_device_eval_batch_size=64,   # batch size for evaluation

    learning_rate=5e-6,
    num_train_epochs=100,  # total # of training epochs
    adam_epsilon=2.5e-9,
    # warmup_steps=100,                # number of warmup steps for learning rate scheduler
    # weight_decay=0.01,
    # no_cuda=True,
)

# No1 team treat dev dataset as eval_dataset, so here I do the same
trainer = Trainer(
    # model=model,                         # the instantiated 🤗 Transformers model to be trained
    model_init=model_init,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()
# trainer.hyperparameter_search()
print("==========================")
trainer.evaluate()
print("==========================")
output = trainer.predict(dev_dataset)
trainer.save_model('results/final')

# with open('bertweet-pred.tsv', 'w') as f:
#     pred = output[0]
#     pred_argmax = pred.argmax(-1)
#     f.write('topic\ttweet_id\tclass_label\trun_id\n')
#     for idx in range(0, len(pred_argmax)):
#         f.write('{}\t{}\t{}\t{}\n'.format(dev_dataset.topic_ids[idx], dev_dataset.ids[idx], pred_argmax[idx], 'test'))

