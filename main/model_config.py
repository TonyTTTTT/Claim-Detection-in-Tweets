from data_preprocess_methods import rewrite_by_GPT, none_operation, split_into_sentences, split_into_frames


# pre-trained weight path provided on [huggingface](https://huggingface.co/models)
# model_path = "vinai/bertweet-covid19-base-uncased"  # model input length limit: 128
# model_path = 'roberta-base'
model_path = 'roberta-large'
# model_path = 'bert-base-uncased'


learning_rate = 2e-6


num_train_epochs = 10
# max_steps = 1350


warm_up_epochs = 1
lr_scheduler_type = "linear"  # available type: ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']


per_device_train_batch_size = 8
device_num = 2


dataset_name = 'CLEF2022_1b'
test_dataset_name = 'CLEF2022_1b'
tags = ['CLEF2022_1b']
run_name = 'baseline'


# seeds that initialize the classifier(Fully Connected Layer)
seeds = [42, 17, 36]
# seeds = [1]


preprocess_function = none_operation
needed_frames_num = 5


do_balancing = False  # specify whether to balancing the class of dataset
do_normalize = True  # specify whether to do rule-based eliminating


# rule-based eliminating
delete_tail = True
delete_at = False
delete_hashtag = True
delete_url = True
delete_emoji = True
delete_tail_punc = False
replace_covid = False
replace_user = False
recover_punc = False
delete_punc = False
delete_dbquote = False
