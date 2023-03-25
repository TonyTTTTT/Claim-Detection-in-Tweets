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
    if os.path.exists('preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(1).f_code.co_name)):
        print('load pkl from preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(1).f_code.co_name))
        with open('preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(1).f_code.co_name),'rb') as f:
            preprocessed_dataset = pickle.load(f)
        return preprocessed_dataset
    return False


def none_operation(ids, topic_ids, texts, labels, dataset):
    return ids, topic_ids, texts, labels


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

    print('write pkl to preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
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

    print('write pkl to preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
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

    print('write pkl to preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
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

    print('write pkl to preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
        pickle.dump([ids_aug, topic_ids_aug, texts_aug, labels_aug], f)

    return ids_aug, topic_ids_aug, texts_aug, labels_aug


def summary_by_GPT(ids, topic_ids, texts, labels, dataset):
    if os.path.exists('preprocess_datasets_tsv/{}_summary_by_GPT_v2.tsv'.format(dataset)):
        data = pd.read_csv('preprocess_datasets_tsv/{}_summary_by_GPT_v2.tsv'.format(dataset), sep='\t')
        ids, topic_ids, texts, labels = read_df_to_lists(data)
        print("load from preprocess_datasets_tsv/{}_summary_by_GPT_v2.tsv".format(dataset))
        return ids, topic_ids, texts, labels

    chatgpt = ChatGPT()
    texts_rewrite = []
    for i in range(0, len(texts)):
        messages = [
            {"role": "system", "content": "You are a summarizer"},
            {"role": "user", "content": texts[i]},
        ]
        res = chatgpt.get_response(messages)

        texts_rewrite.append(res)

    df = pd.DataFrame(list(zip(topic_ids, ids, texts_rewrite, labels)), columns=['topic', 'tweet_id', 'tweet_text', 'class_label'])
    df.to_csv('preprocess_datasets_tsv/{}_summary_by_GPT_v2.tsv'.format(dataset), sep='\t', index=False)

    return ids, topic_ids, texts_rewrite, labels


def concate_all_frames(ids, topic_ids, texts, labels, dataset):
    '''
    extract all frames, cocate them and return
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
            frame_concate = ''
            res = discard_similar_frame(res)

            for frame in res:
                frame_concate += frame+'. '

            frame_concate = frame_concate.strip()
            texts_aug.append(frame_concate)
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            labels_aug.append(labels[i])

    print('write pkl to preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
        pickle.dump([ids_aug, topic_ids_aug, texts_aug, labels_aug], f)

    return ids_aug, topic_ids_aug, texts_aug, labels_aug


def too_similar(frame_short, frame_long):
    frame_short_list = frame_short.split(' ')
    frame_long_list = frame_long.split(' ')
    overlap = [word for word in frame_short_list if word in frame_long_list]
    overlap_rate = len(overlap) / len(frame_short_list)

    if overlap_rate > 0.5:
        return True
    else:
        return False


def len_of_frame(frame):
    return len(frame[0])


def order_of_frame(frame):
    return frame[1]


def discard_similar_frame(frames):
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
            if frame_cnt >= 2:
                break
            frames_reduced.append(frame)
            frame_cnt += 1

    frames_reduced.sort(key=order_of_frame)

    for i in range(0, len(frames_reduced)):
        frames_reduced[i] = frames_reduced[i][0]

    return frames_reduced


if __name__ == '__main__':
    ids = [23423, 1161]
    topic_ids = ['pig', 'cat']
    texts = ["India 's gift of 100,000 COVID-19 vaccines arrived Barbados earlier today. This was a very special moment for all Barbadians and I want to thank Prime Minister Modi for his quick, decisive, and magnanimous action in allowing us to be the beneficiary of these vaccines. HTTPURL",
             "We donât yet have all the tools we need to fight COVID-19. This is an important step toward having treatments, while we also explore vaccines and diagnostics. Thanks to @wellcometrust and @mastercard for launching this effort with us. https://t.co/M8AJ3083zK"]
    labels = [0, 1]
    dataset = 'GGG'
    ids_aug, topic_ids_aug, texts_aug, labels_aug = summary_by_GPT(ids, topic_ids, texts, labels, dataset)
