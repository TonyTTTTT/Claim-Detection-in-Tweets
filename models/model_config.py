from data_preprocess_methods import insert_srl_tag, extract_to_sentence_level, extract_all_frames, none_operation, \
    concate_all_frames, convert_to_srl_tag, rewrite_by_GPT


# tokenizer不同會導致encoding長度(tokens個數)不一樣 (應該是vocabulary的問題)

# model input length limit: 128
# model_path = "vinai/bertweet-covid19-base-uncased"

# size of GPT3: 175 billion

# size: 123 million
model_path = 'roberta-base'

# size: 354 million
# model_path = 'roberta-large'

# model_path = 'bert-base-uncased'

learning_rate = 3e-5
num_train_epochs = 50
warm_up_epochs = 5

# ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
lr_scheduler_type = "linear"
per_device_train_batch_size = 4

dataset = 'rewrite_to_understandable_by_GPT'

preprocess_function = concate_all_frames
do_normalize = False
