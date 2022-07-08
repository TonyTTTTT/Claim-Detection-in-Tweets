from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from data_loader import DataLoader, compute_metrics
from data_preprocess_methods import insert_srl_tag, extract_to_sentence_level

model_path = "vinai/bertweet-covid19-base-uncased"
# model_path = 'roberta-base'

dataloader = DataLoader(preprocess_function=insert_srl_tag)
train_dataset, test_dataset = dataloader.get_dataset()


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
    return model


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    logging_dir='./logs',  # directory for storing logs
    evaluation_strategy='epoch',
    logging_steps=100,
    save_steps=100,
    per_device_train_batch_size=4,  # batch size per device during training
    # per_device_eval_batch_size=64,   # batch size for evaluation

    # learning_rate=5e-4,
    # num_train_epochs=5,  # total # of training epochs
    adam_epsilon=2.5e-9,
    # warmup_steps=100,                # number of warmup steps for learning rate scheduler
    # weight_decay=0.01,
    # no_cuda=True,
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
trainer.evaluate()
print("==========================")
output = trainer.predict(test_dataset)
trainer.save_model('results/final')

with open('add-srl-longest-tag-64-bertweet-pred-test.tsv', 'w') as f:
    pred = output[0]
    pred_argmax = pred.argmax(-1)
    # score_sum = 0
    # frame_cnt = 0
    for idx in range(0, len(pred)):
        # if idx != 0 and test_dataset.ids[idx] != test_dataset.ids[idx-1]:
        #     score_avg = score_sum/frame_cnt
        #     f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score_avg, 'test'))
        #     score_sum = 0
        #     frame_cnt = 0
        score = pred[idx][pred_argmax[idx]]
        if pred_argmax[idx] == 0:
            # if the score of label0 is bigger, that it be negative, for create final score
            score = -1 * score
        f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score, 'test'))
        # score_sum += score
        # frame_cnt += 1
        # print('{}\t{}\t{}\t{}'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score, 'test'))

    # f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score_avg, 'test'))
