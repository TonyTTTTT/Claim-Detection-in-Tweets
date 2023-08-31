from data_loader import DataLoader, compute_metrics
from transformers import RobertaForSequenceClassification, Trainer
from data_preprocess_methods import split_into_frames
from model_config import *
from transformers import AutoTokenizer
import torch
from model_training import calculate_article_score_from_sentence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mine_model_path = 'weights/{}_{}_{}'.format(dataset_name, test_dataset_name, run_name)
print('===========\nmodel path: {}\n==========='.format(mine_model_path))
model = RobertaForSequenceClassification.from_pretrained(mine_model_path)
model.eval()


dataloader_frame = DataLoader(preprocess_function=none_operation, dataset=test_dataset_name,
                              do_normalize=do_normalize, needed_frames_num=needed_frames_num,
                              do_balancing=do_balancing)
train_dataset_frame, dev_dataset_frame, test_dataset_frame = dataloader_frame.get_dataset(include_test=True)
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    # model_init=model_init,
    # args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset_frame,         # training datasets
    eval_dataset=test_dataset_frame,            # evaluation datasets
    compute_metrics=compute_metrics
)

output_frame = trainer.predict(test_dataset_frame)
print('\nnone operation:\n======================\nf1_macro: {}\nf1: {}\naccuracy: {}\nconfusion matrix: {}\n========================='.format(output_frame.metrics['test_f1_macro'], output_frame.metrics['test_f1'], output_frame.metrics['test_accuracy'], output_frame.metrics['test_confusion_matrix']))


dataloader_frame = DataLoader(preprocess_function=split_into_frames, dataset=test_dataset_name,
                              do_normalize=do_normalize, needed_frames_num=needed_frames_num,
                              do_balancing=do_balancing)
train_dataset_frame, dev_dataset_frame, test_dataset_frame = dataloader_frame.get_dataset(include_test=True)


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    # model_init=model_init,
    # args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset_frame,         # training datasets
    eval_dataset=test_dataset_frame,            # evaluation datasets
    compute_metrics=compute_metrics
)

output_frame = trainer.predict(test_dataset_frame)
# print(output_frame)
print('\nsplit by SRL:')
split_into_frames_f1_macro, split_into_frames_f1, split_into_frames_acc, split_into_frames_confusionMatrix, split_into_frames_wrong_predicted_idx = calculate_article_score_from_sentence(test_dataset_frame, output_frame, 'max')



# text = 'China will admit coronavirus coming from its P4 lab BioWeapon.'
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, normalization=False)
# tokenizer.model_max_length = 128
# text_encoding = tokenizer(text, truncation=True, padding='max_length')
# input_tensor = torch.tensor(text_encoding['input_ids']).resize(1, 128).to(device)
#
# print(text)
# print(model(input_tensor))

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