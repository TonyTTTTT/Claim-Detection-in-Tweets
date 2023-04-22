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
    if os.path.exists('preprocess_datasets/{}.pkl'.format(dataset)):
        print('======================\nload pkl from preprocess_datasets/{}.pkl'
              '\n======================='.format(dataset))
        with open('preprocess_datasets/{}.pkl'.format(dataset), 'rb') as f:
            preprocessed_dataset = pickle.load(f)
        return preprocessed_dataset
    return False


def none_operation(*args):
    ids = args[0]
    topic_ids = args[1]
    texts = args[2]
    labels = args[3]
    return ids, topic_ids, texts, labels


def concate_frames(*args):
    '''
    extract all frames, cocate them and return
    '''
    ids = args[0]
    topic_ids = args[1]
    texts = args[2]
    labels = args[3]
    dataset = args[4]
    part = args[5]
    concate_frames_num = args[6]

    preprocess_dataset_name = '{}_top_{}_{}'.format(dataset, concate_frames_num, part)

    preprocessed_dataset = check_if_exist(preprocess_dataset_name)
    if preprocessed_dataset:
        return preprocessed_dataset[0], preprocessed_dataset[1], preprocessed_dataset[2], preprocessed_dataset[3]

    predictor = SRLPredictor()
    ids_aug = []
    topic_ids_aug = []
    texts_aug = []
    labels_aug = []

    print('=======================\nExtracting SRL frames\n========================')
    for i in range(0, len(texts)):
        res = predictor.get_frames(texts[i])
        if res == []:
            texts_aug.append(texts[i])
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            labels_aug.append(labels[i])
        else:
            frame_concate = ''
            res = discard_similar_frame(res, concate_frames_num)

            for frame in res:
                frame_concate += frame+'. '

            frame_concate = frame_concate.strip()
            texts_aug.append(frame_concate)
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            labels_aug.append(labels[i])

    print('=====================\nwrite pkl to preprocess_datasets/{}.pkl'
          '\n============================='.format(preprocess_dataset_name))
    with open('preprocess_datasets/{}.pkl'.format(preprocess_dataset_name), 'wb') as f:
        pickle.dump([ids_aug, topic_ids_aug, texts_aug, labels_aug], f)

    return ids_aug, topic_ids_aug, texts_aug, labels_aug


def too_similar(frame_short, frame_long):
    frame_short_list = frame_short.split(' ')
    frame_long_list = frame_long.split(' ')
    overlap = [word for word in frame_short_list if word in frame_long_list]
    overlap_rate = len(overlap) / len(frame_short_list)

    if overlap_rate > 0.7:
        return True
    else:
        return False


def len_of_frame(frame):
    return len(frame[0])


def order_of_frame(frame):
    return frame[1]


def discard_similar_frame(frames, concate_frames_num):
    frames_reduced = []

    for i in range(0, len(frames)):
        frames[i] = [frames[i], i]

    frames.sort(key=len_of_frame, reverse=True)

    frame_cnt = 0
    for frame in frames:
        add_flag = True
        for frame_reduced in frames_reduced:
            add_flag = not too_similar(frame[0], frame_reduced[0])
            if not add_flag:
                break

        if add_flag:
            if frame_cnt >= concate_frames_num:
                break
            frames_reduced.append(frame)
            frame_cnt += 1

    frames_reduced.sort(key=order_of_frame)

    for i in range(0, len(frames_reduced)):
        frames_reduced[i] = frames_reduced[i][0]

    return frames_reduced


def rewrite_by_GPT(*args):
    ids = args[0]
    topic_ids = args[1]
    texts = args[2]
    labels = args[3]
    dataset = args[4]
    part = args[5]

    rewrite_method = 'simplify_by_GPT'
    preprocess_dataset_name = '{}_{}_{}'.format(dataset, rewrite_method, part)

    if os.path.exists('preprocess_datasets_tsv/{}.tsv'.format(preprocess_dataset_name)):
        data = pd.read_csv('preprocess_datasets_tsv/{}.tsv'.format(preprocess_dataset_name), sep='\t')
        ids, topic_ids, texts, labels = read_df_to_lists(data)
        print("===================\nload from preprocess_datasets_tsv/{}.tsv"
              "\n=====================".format(preprocess_dataset_name))
        return ids, topic_ids, texts, labels

    chatgpt = ChatGPT()
    texts_rewrite = []
    for i in range(0, len(texts)):
        messages = [
            # {"role": "system", "content": "Can you rephrase the following article to be more clear and easy to understand?"},
            {"role": "user", "content": texts[i] + '\nsimplify:'}
        ]
        res = chatgpt.get_response(messages)
        # res = res.split('\n')[0]
        # print('res split by next line: {}'.format(res))
        texts_rewrite.append(res)

    df = pd.DataFrame(list(zip(topic_ids, ids, texts_rewrite, labels)), columns=['topic', 'tweet_id', 'tweet_text', 'class_label'])
    df['tweet_id'] = df['tweet_id'].astype(str)
    print('=====================\nwrite tsv to preprocess_datasets_tsv/{}.pkl'
          '\n============================='.format(preprocess_dataset_name))
    df.to_csv('preprocess_datasets_tsv/{}.tsv'.format(preprocess_dataset_name), sep='\t', index=False)

    return ids, topic_ids, texts_rewrite, labels


if __name__ == '__main__':
    ids = [23423, 1161]
    topic_ids = ['pig', 'cat']
    texts = ["India has sent 100,000 doses of COVID-19 vaccines to Barbados, and they arrived earlier today. This is a significant and meaningful gesture, and the people of Barbados are grateful to the Prime Minister of India, Mr. Modi, for his prompt and generous decision to send these vaccines. Thank you, Mr. Modi.",
             "Being a part of @ETHPnews, we are delighted to announce that we have established two #COVID19 Immunization Clinics in #EastToronto. This week, these clinics will provide vaccines to eligible priority groups, such as health care workers and individuals over 80 years old. For more information, please visit: https://t.co/t890KePvBG https://t.co/We2EdhFitS."]
    labels = [0, 1]
    dataset = 'GGG'
    part = 'train'
    concate_frames_num = 5

    ids_aug, topic_ids_aug, texts_aug, labels_aug = concate_frames(ids, topic_ids, texts, labels, dataset,
                                                                   part, concate_frames_num)
