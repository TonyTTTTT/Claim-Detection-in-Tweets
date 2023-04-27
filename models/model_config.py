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
# learning_rate = 1e-6  # clef2022 1b server
learning_rate = 5e-7  # clef2022 1b local

# num_train_epochs = 10  # clef2022 1b server
num_train_epochs = 5  # clef2022 1b local

warm_up_epochs = 1

# ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
lr_scheduler_type = "constant_with_warmup"
per_device_train_batch_size = 4
device_num = 1

dataset = 'CLEF2022_1b'

preprocess_function = none_operation
concate_frames_num = 5
do_normalize = True
