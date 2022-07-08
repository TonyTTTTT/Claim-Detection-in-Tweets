from srl_predictor import SRLPredictor


def none_operation(ids, topic_ids, texts, labels):
    return ids, topic_ids, texts, labels


def insert_srl_tag(ids, topic_ids, texts, labels):
    predictor = SRLPredictor()
    ids_aug = []
    topic_ids_aug = []
    texts_aug = []
    labels_aug = []

    for i in range(0, len(texts)):
        res = predictor.get_frames_tag(texts[i])
        if res:
            longest_idx = 0
            for j in range(1, len(res)):
                if len(res[j]) > len(res[longest_idx]):
                    longest_idx = j
            ids_aug.append(ids[i])
            topic_ids_aug.append(topic_ids[i])
            inserted_text = texts[i] + res[longest_idx]
            texts_aug.append(inserted_text)
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

    return ids_aug, topic_ids_aug, texts_aug, labels_aug


def extract_to_sentence_level(ids, topic_ids, texts, labels):
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

    return ids_aug, topic_ids_aug, texts_aug, labels_aug
