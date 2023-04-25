import os.path
import pickle
import sys
from srl_predictor import SRLPredictor
from chatGPT_api import ChatGPT
import pandas as pd


def read_df_to_lists(data):
    texts = []
    labels = []
    ids = []
    topic_ids = []

    if data.columns[-1] != 'class_label' and data.columns[-1] != 'check_worthiness':
        for row in data.iterrows():
            # get the tweet_text field
            texts.append(row[1].values[-1])
            # get teh check_worthiness field
            topic_ids.append(row[1].values[0])
            ids.append(row[1].values[1])
        return ids, topic_ids, texts

    for row in data.iterrows():
        # get the tweet_text field
        texts.append(row[1].values[-2])
        # get teh check_worthiness field
        labels.append(row[1].values[-1])
        topic_ids.append(row[1].values[0])
        ids.append(row[1].values[1])
    return ids, topic_ids, texts, labels


def check_if_exist(dataset):
    if os.path.exists('preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(1).f_code.co_name)):
        print('load pkl from preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(1).f_code.co_name))
        with open('preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(1).f_code.co_name),'rb') as f:
            preprocessed_dataset = pickle.load(f)
        return preprocessed_dataset
    return False


def convert_to_srl_tag(ids, topic_ids, texts, labels, dataset):
    '''
    srl tags for every frames
    '''
    preprocessed_dataset = check_if_exist(dataset)
    if preprocessed_dataset:
        return preprocessed_dataset[0], preprocessed_dataset[1], preprocessed_dataset[2], preprocessed_dataset[3]

    predictor = SRLPredictor()
    ids_aug = []
    topic_ids_aug = []
    texts_aug = []
    labels_aug = []

    for i in range(0, len(texts)):
        res = predictor.get_frames_tag(texts[i])
        if res:
            # longest_idx = 0
            # for j in range(1, len(res)):
            #     if len(res[j]) > len(res[longest_idx]):
            #         longest_idx = j
            all_frmaes_tags = ''
            for frame in res:
                all_frmaes_tags += ' '+frame
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            texts_aug.append(all_frmaes_tags)
            labels_aug.append(labels[i])
        else:
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            texts_aug.append(texts[i])
            labels_aug.append(labels[i])

    print('write pkl to preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
        pickle.dump([ids_aug, topic_ids_aug, texts_aug, labels_aug], f)

    return ids_aug, topic_ids_aug, texts_aug, labels_aug


def insert_srl_tag(ids, topic_ids, texts, labels, dataset):
    '''
    append srl tag to the end of original texts
    '''
    preprocessed_dataset = check_if_exist(dataset)
    if preprocessed_dataset:
        return preprocessed_dataset[0], preprocessed_dataset[1], preprocessed_dataset[2], preprocessed_dataset[3]

    predictor = SRLPredictor()
    ids_aug = []
    topic_ids_aug = []
    texts_aug = []
    labels_aug = []

    for i in range(0, len(texts)):
        res = predictor.get_frames_tag(texts[i])
        if res:
            # longest_idx = 0
            # for j in range(1, len(res)):
            #     if len(res[j]) > len(res[longest_idx]):
            #         longest_idx = j
            inserted_text = texts[i]
            for frame in res:
                inserted_text += ' '+frame
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            texts_aug.append(inserted_text)
            labels_aug.append(labels[i])
        else:
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            texts_aug.append(texts[i])
            labels_aug.append(labels[i])

    print('write pkl to preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
        pickle.dump([ids_aug, topic_ids_aug, texts_aug, labels_aug], f)

    return ids_aug, topic_ids_aug, texts_aug, labels_aug


def extract_to_sentence_level(ids, topic_ids, texts, labels, dataset):
    '''
    extract the longest frame as new data instance
    '''
    preprocessed_dataset = check_if_exist(dataset)
    if preprocessed_dataset:
        return preprocessed_dataset[0], preprocessed_dataset[1], preprocessed_dataset[2], preprocessed_dataset[3]

    predictor = SRLPredictor()
    ids_aug = []
    topic_ids_aug = []
    texts_aug = []
    labels_aug = []

    for i in range(0, len(texts)):
        res = predictor.get_frames(texts[i])
        if res != []:
            longest_idx = 0
            for j in range(1, len(res)):
                if len(res[j]) > len(res[longest_idx]):
                    longest_idx = j
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            texts_aug.append(res[longest_idx])
            labels_aug.append(labels[i])
            # for frame in res:
            #     ids_aug.append(ids[i])
            #     topic_ids_aug.append(topic_ids[i])
            #     texts_aug.append(frame)
            #     labels_aug.append(labels[i])
        else:
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            texts_aug.append(texts[i])
            labels_aug.append(labels[i])

    print('write pkl to preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
        pickle.dump([ids_aug, topic_ids_aug, texts_aug, labels_aug], f)

    return ids_aug, topic_ids_aug, texts_aug, labels_aug


def extract_all_frames(ids, topic_ids, texts, labels, dataset):
    '''
    extract and return all frames
    '''
    preprocessed_dataset = check_if_exist(dataset)
    if preprocessed_dataset:
        return preprocessed_dataset[0], preprocessed_dataset[1], preprocessed_dataset[2], preprocessed_dataset[3]

    predictor = SRLPredictor()
    ids_aug = []
    topic_ids_aug = []
    texts_aug = []
    labels_aug = []

    for i in range(0, len(texts)):
        res = predictor.get_frames(texts[i])
        if res == []:
            texts_aug.append(texts[i])
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            labels_aug.append(labels[i])
        else:
            for frame in res:
                texts_aug.append(frame)
                ids_aug.append(ids[i])
                topic_ids_aug.append(topic_ids[i])
                labels_aug.append(labels[i])

    print('write pkl to preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets_SRL/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
        pickle.dump([ids_aug, topic_ids_aug, texts_aug, labels_aug], f)

    return ids_aug, topic_ids_aug, texts_aug, labels_aug


if __name__ == '__main__':
    ids = [23423, 1161]
    topic_ids = ['pig', 'cat']
    texts = ["India has sent 100,000 doses of COVID-19 vaccines to Barbados, and they arrived earlier today. This is a significant and meaningful gesture, and the people of Barbados are grateful to the Prime Minister of India, Mr. Modi, for his prompt and generous decision to send these vaccines. Thank you, Mr. Modi.",
             "We donât yet have all the tools we need to fight COVID-19. This is an important step toward having treatments, while we also explore vaccines and diagnostics. Thanks to @wellcometrust and @mastercard for launching this effort with us. https://t.co/M8AJ3083zK"]
    labels = [0, 1]
    dataset = 'GGG'
    ids_aug, topic_ids_aug, texts_aug, labels_aug = extract_all_frames(ids, topic_ids, texts, labels, dataset)
