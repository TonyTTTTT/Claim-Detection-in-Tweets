from data_preprocess_methods import insert_srl_tag, extract_to_sentence_level, extract_all_frames, none_operation, \
    concate_all_frames, convert_to_srl_tag


# tokenizer不同會導致encoding長度(tokens個數)不一樣 (應該是vocabulary的問題)

# model input length limit: 128
model_path = "vinai/bertweet-covid19-base-uncased"

# model_path = 'roberta-base'

# model_path = 'roberta-large'

# model_path = 'bert-base-uncased'

learning_rate = 5e-5
num_train_epochs = 10
warm_up_epochs = 2

# ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
lr_scheduler_type = "linear"
per_device_train_batch_size = 1

dataset = 'CLEF2022'

preprocess_function = none_operation
