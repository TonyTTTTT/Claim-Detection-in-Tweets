from data_loader import DataLoader, compute_metrics
from transformers import AutoConfig, AutoModel
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from data_preprocess_methods import split_into_sentences, split_into_frames
from model_config import *
from transformers import AutoTokenizer
import torch


mine_model_path = 'results/{}_{}_{}'.format(dataset_name, test_dataset_name, run_name)

# dataloader_frame = DataLoader(preprocess_function=split_into_frames, dataset=test_dataset_name,
#                               do_normalize=do_normalize, concate_frames_num=concate_frames_num,
#                               do_balancing=do_balancing)
# train_dataset_frame, dev_dataset_frame, test_dataset_frame = dataloader_frame.get_dataset(include_test=True)
# config = AutoConfig.from_pretrained(mine_model_path)
# model = AutoModel.from_config(config)

model = RobertaForSequenceClassification.from_pretrained(mine_model_path)
model.eval()

text = 'China will admit coronavirus coming from its P4 lab BioWeapon.'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, normalization=False)
tokenizer.model_max_length = 128
text_encoding = tokenizer(text, truncation=True, padding='max_length')
input_tensor = torch.tensor(text_encoding['input_ids']).resize(1, 128)

print(mine_model_path)
print(text)
print(model(input_tensor))
# trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     # model_init=model_init,
#     # args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=test_dataset,            # evaluation dataset
#     compute_metrics=compute_metrics
# )
#
# output = trainer.predict(test_dataset)
# print(output)

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