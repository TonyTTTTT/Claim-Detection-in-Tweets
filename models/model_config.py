from data_preprocess_methods import insert_srl_tag, extract_to_sentence_level, extract_all_frames, none_operation, \
    concate_all_frames, convert_to_srl_tag


# tokenizer不同會導致encoding長度(tokens個數)不一樣 (應該是vocabulary的問題)

# model input length limit: 128
model_path = "vinai/bertweet-covid19-base-uncased"

# model_path = 'roberta-base'

# model_path = 'bert-base-uncased'


num_train_epochs = 15

dataset = 'CLEF2022'

preprocess_function = none_operation
