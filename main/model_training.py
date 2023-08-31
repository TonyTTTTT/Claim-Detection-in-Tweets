from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, AutoConfig,\
    BertForSequenceClassification
from data_loader import DataLoader, compute_metrics
from model_config import *
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
from data_preprocess_methods import split_into_sentences, split_into_frames, none_operation
import wandb

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
    preds = predictions.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    confusionMatrix = confusion_matrix(labels, preds).tolist()
    f1_macro = f1_score(labels, preds, average='macro')
    agree = preds == labels
    wrong_predicted_idx = np.where(agree == False)[0].tolist()
    print('=================================\n'
          'using method: {}\n'
          'f1_macro at article-level: {}\nf1 at article-level: {}\nacc at article-level: {}\nconfusion matrix: {}'
          '\n=================================\n'.format(combine_method, f1_macro, f1, acc, confusionMatrix))

    return f1_macro, f1, acc, confusionMatrix, wrong_predicted_idx


def model_init():
    configuration = AutoConfig.from_pretrained(model_path)

    if model_path.startswith('roberta') or 'bertweet' in model_path:
        model = RobertaForSequenceClassification.from_pretrained(model_path, config=configuration)
    elif model_path.startswith('bert'):
        model = BertForSequenceClassification.from_pretrained(model_path, config=configuration)

    return model


if __name__ == '__main__':
    dataloader = DataLoader(preprocess_function=preprocess_function, dataset=dataset_name, do_normalize=do_normalize,
                            needed_frames_num=needed_frames_num, do_balancing=do_balancing)
    train_dataset_train, train_dataset_dev, train_dataset_test = dataloader.get_dataset(include_test=True)


    training_args = TrainingArguments(
        output_dir='weights',  # model save dir
        logging_dir='./logs/{}/{}_{}_{}_{}'.format(dataset_name, model_path, dataloader.preprocess_function.__name__, lr_scheduler_type, num_train_epochs),  # directory for storing logs
        evaluation_strategy='epoch',
        # logging_steps=100,
        logging_strategy='epoch',

        # save_steps=10000,
        save_strategy="no",
        # save_total_limit=1,
        # load_best_model_at_end=True,

        logging_first_step=True,

        per_device_train_batch_size=per_device_train_batch_size,
        # per_device_eval_batch_size=64,

        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        # max_steps=max_steps,
        # adam_epsilon=2.5e-9,
        warmup_steps=(len(train_dataset_train.ids) / (per_device_train_batch_size * device_num)) * warm_up_epochs,
        # weight_decay=0,
        # no_cuda=True,
        lr_scheduler_type=lr_scheduler_type,
        # seed=42,
    )

    f1_macro_sum = 0
    f1_macro_sen_sum = 0
    f1_macro_frame_sum = 0
    acc_sum = 0
    acc_sen_sum = 0
    acc_frame_sum = 0
    f1_sum = 0
    f1_sen_sum = 0
    f1_frame_sum = 0

    tags_last_run = tags.copy()
    tags_last_run.append('last run')

    for i in range(0, len(seeds)):
        if i == len(seeds)-1:
            tags_use = tags_last_run
        else:
            tags_use = tags

        run = wandb.init(
            project="Claim Detection in Tweets",
            name='{}_{}'.format(run_name, i),
            tags=tags_use
        )

        training_args.seed = seeds[i]
        # No1 team treat dev datasets as eval_dataset, so here I do the same
        trainer = Trainer(
            model_init=model_init,
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset_train,         # training datasets
            eval_dataset=train_dataset_test,            # evaluation datasets
            compute_metrics=compute_metrics
        )

        trainer.train()
        # trainer.hyperparameter_search()
        print("==========================")
        result = trainer.evaluate()
        print("==========================")
        output = trainer.predict(train_dataset_test)
        print("==========================")

        f1_sum += result['eval_f1']
        acc_sum += result['eval_accuracy']
        f1_macro_sum += result['eval_f1_macro']




        if test_dataset_name == 'same':
            testing_dataset_name = dataloader.preprocess_dataset_name

        dataloader_sentence = DataLoader(preprocess_function=split_into_sentences, dataset=test_dataset_name,
                                         do_normalize=do_normalize, needed_frames_num=needed_frames_num, do_balancing=do_balancing)
        train_dataset_sentence, dev_dataset_sentence, test_dataset_sentence = dataloader_sentence.get_dataset(include_test=True)
        output_sentence = trainer.predict(test_dataset_sentence)
        split_into_sentences_f1_macro, split_into_sentences_f1, split_into_sentences_acc, split_into_sentences_confusionMatrix, split_into_sentences_wrong_predicted_idx = calculate_article_score_from_sentence(test_dataset_sentence, output_sentence, 'max')
        wandb.log({"f1_macro_split_to_sentences": split_into_sentences_f1_macro, "f1_split_to_sentences": split_into_sentences_f1, "acc_split_to_sentences": split_into_sentences_acc})
        f1_macro_sen_sum += split_into_sentences_f1_macro
        f1_sen_sum += split_into_sentences_f1
        acc_sen_sum += split_into_sentences_acc

        dataloader_frame = DataLoader(preprocess_function=split_into_frames, dataset=test_dataset_name,
                                      do_normalize=do_normalize, needed_frames_num=needed_frames_num, do_balancing=do_balancing)
        train_dataset_frame, dev_dataset_frame, test_dataset_frame = dataloader_frame.get_dataset(include_test=True)
        output_frame = trainer.predict(test_dataset_frame)
        split_into_frames_f1_macro, split_into_frames_f1, split_into_frames_acc, split_into_frames_confusionMatrix, split_into_frames_wrong_predicted_idx = calculate_article_score_from_sentence(test_dataset_frame, output_frame, 'max')
        wandb.log({"f1_macro_split_to_frames": split_into_frames_f1_macro, "f1_split_to_frames": split_into_frames_f1, "acc_split_to_frames": split_into_frames_acc})
        f1_macro_frame_sum += split_into_frames_f1_macro
        f1_frame_sum += split_into_frames_f1
        acc_frame_sum += split_into_frames_acc


        if i < len(seeds)-1:
            run.finish()

    # columns = ["wrong_predicted_idx"]
    # Method 1
    # data =
    # table = wandb.Table(data=data, columns=columns)
    # wandb.log({"examples": table})

    dataloader_sentence.write_prediciton_to_file(output_sentence.predictions.argmax(axis=-1), dataset_name, test_dataset_name, 'sentence')
    dataloader_frame.write_prediciton_to_file(output_frame.predictions.argmax(axis=-1), dataset_name, test_dataset_name, 'frame')

    if test_dataset_name == dataset_name:
        test_dataset_test = train_dataset_test
    else:
        dataloader_test = DataLoader(preprocess_function=none_operation, dataset=test_dataset_name,
                                     do_normalize=do_normalize, needed_frames_num=needed_frames_num, do_balancing=do_balancing)
        test_dataset_train, test_dastaset_dev, test_dataset_test = dataloader_test.get_dataset(include_test=True)

    with open('wrong_predictions_idx/{}_{}_{}.txt'.format(dataset_name, test_dataset_name, run_name), 'w') as f:
        f.write('origin:\n{}\n{}\n'.format(str(output[2]['test_confusion_matrix']), str(np.array(train_dataset_test.ids)[output[2]['test_wrong_predicted_idx']].astype(int).tolist())))
        f.write('split to sentence:\n{}\n{}\n'.format(str(split_into_sentences_confusionMatrix), str(np.array(test_dataset_test.ids)[split_into_sentences_wrong_predicted_idx].astype(int).tolist())))
        f.write('split to frames:\n{}\n{}\n'.format(str(split_into_frames_confusionMatrix), str(np.array(test_dataset_test.ids)[split_into_frames_wrong_predicted_idx].astype(int).tolist())))

    f1_sum /= len(seeds)
    f1_sen_sum /= len(seeds)
    f1_frame_sum /= len(seeds)
    acc_sum /= len(seeds)
    acc_sen_sum /= len(seeds)
    acc_frame_sum /= len(seeds)
    f1_macro_sum /= len(seeds)
    f1_macro_sen_sum /= len(seeds)
    f1_macro_frame_sum /= len(seeds)

    print("f1_macro_avg: {}, f1_avg: {}, acc_avg: {}\nf1_macro_sen_avg: {}, f1_sen_avg: {}, acc_sen_avg: {}\nf1_macro_frame_avg: {}, f1_frame_avg: {}, acc_frame_avg: {}".format(f1_macro_sum, f1_sum, acc_sum, f1_macro_sen_sum,f1_sen_sum, acc_sen_sum, f1_macro_frame_sum, f1_frame_sum, acc_frame_sum))
    wandb.log({"f1_macro_avg": f1_macro_sum, "f1_avg": f1_sum, "acc_avg": acc_sum, "f1_macro_sen_avg": f1_macro_sen_sum, "f1_sen_avg": f1_sen_sum, "acc_sen_avg": acc_sen_sum, "f1_macro_frame_avg": f1_macro_frame_sum, "f1_frame_avg": f1_frame_sum, "acc_frame_avg": acc_frame_sum})
    run.finish()

    trainer.save_model('weights/{}_{}_{}'.format(dataset_name, test_dataset_name, run_name))
