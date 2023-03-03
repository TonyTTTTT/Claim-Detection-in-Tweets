from data_loader import DataLoader, compute_metrics
from transformers import AutoConfig, AutoModel
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from data_preprocess_methods import *


model_path = "results/mine_best/final"

data_loader = DataLoader(preprocess_function=none_operation)
train_dataset, dev_dataset, test_dataset = data_loader.get_dataset(include_test=True)
# config = AutoConfig.from_pretrained("./results/mine_best/final")
# model = AutoModel.from_config(config)
model = RobertaForSequenceClassification.from_pretrained(model_path)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    # model_init=model_init,
    # args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)

output = trainer.predict(test_dataset)
print(output)

# with open('bertweet-pred.tsv', 'w') as f:
#     pred = output[0]
#     pred_argmax = pred.argmax(-1)
#     for idx in range(0, len(pred)):
#         score = pred[idx][pred_argmax[idx]]
#         if pred_argmax[idx] == 0:
#             # if the score of label0 is bigger, that it be negative, for create final score
#             score = -1 * score
#         # print('{}\t{}\t{}\t{}'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score, 'test'))
#         f.write('{}\t{}\t{}\t{}\n'.format(test_dataset.topic_ids[idx], test_dataset.ids[idx], score, 'test'))