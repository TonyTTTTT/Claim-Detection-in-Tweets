from data_preprocess_methods import none_operation, concate_frames, rewrite_by_GPT


# tokenizer不同會導致encoding長度(tokens個數)不一樣 (應該是vocabulary的問題)

# model input length limit: 128
# model_path = "vinai/bertweet-covid19-base-uncased"

# size of GPT3: 175 billion

# size: 123 million
model_path = 'roberta-base'

# size: 354 million
# model_path = 'roberta-large'

# model_path = 'bert-base-uncased'


# learning_rate = 3e-5  # clef2022 1a
# learning_rate = 1e-6  # server
learning_rate = 5e-7  # local

# num_train_epochs = 10  # LESA
# num_train_epochs = 7 # CLEF2022 1b
# num_train_epochs = 5 # other sentence-level
num_train_epochs = 1  # local

warm_up_epochs = 1

# ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
lr_scheduler_type = "constant_with_warmup"
per_device_train_batch_size = 4
device_num = 1

dataset_name = 'CLEF2022_1b'
test_dataset_name = 'LESA'
tags = ['sentence level']
run_name = 'baseline'

seeds = [42, 17, 36]
preprocess_function = rewrite_by_GPT
concate_frames_num = 5
do_normalize = False
do_balancing = False
