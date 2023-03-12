import os.path
import pickle
import sys
from srl_predictor import SRLPredictor


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
            for frame in res:
                frame_concate += frame+'. '
            texts_aug.append(frame_concate)
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            labels_aug.append(labels[i])

    print('write pkl to preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name))
    with open('preprocess_datasets/{}_{}.pkl'.format(dataset, sys._getframe(0).f_code.co_name), 'wb') as f:
        pickle.dump([ids_aug, topic_ids_aug, texts_aug, labels_aug], f)

    return ids_aug, topic_ids_aug, texts_aug, labels_aug


if __name__ == '__main__':
    ids = [1]
    topic_ids = ['pig']
    texts = ['Just watched half an hour of @CNN coverage of massive nationwide protests. There was no mention of '
             'coronavirus or need for social distancing. Literally zero. The pandemic hasn’t gone away. '
             'Well over 100,000 have died. There’s no vaccine. We have to do better on this.']
    labels = [0]
    dataset = 'GGG'
    ids_aug, topic_ids_aug, texts_aug, labels_aug = convert_to_srl_tag(ids, topic_ids, texts, labels, dataset)
