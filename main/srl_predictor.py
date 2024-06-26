import abc
import numpy as np
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor


class SRLPredictor():
    def __init__(self):
        self.srl_predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
        )
        self.srl_predictor._model = self.srl_predictor._model.cuda()

    def compute(self, sample, claim_db, k, encoder, prepend_title_sentence=True, prepend_title_frame=True):

        if prepend_title_sentence and "title" in sample:
            sentence = sample["title"] + " " + sample["sentence"]
        else:
            sentence = sample["sentence"]

        sentence_embedding = encoder.encode(sentence)
        scores = claim_db.get_scores(sentence_embedding)

        # Get Frames
        if "frames" not in sample:
            sample["frames"] = self.get_frames(sample)

        frames = sample["frames"]
        for frame in frames:

            if prepend_title_frame and "title" in sample:
                frame = sample["title"] + " " + frame

            frame_embedding = encoder.encode(frame)
            similarity = claim_db.get_scores(frame_embedding)
            scores = np.maximum(scores, similarity)

        top_claim_index = self.get_top_claim_index(scores, k)

        return top_claim_index

    def get_frames(self, sample):
        frames = []
        try:
            res = self.srl_predictor.predict(
                sentence=sample
            )
            words = res["words"]
            for frame in res["verbs"]:
                buffer = []

                arg_cnt = len([i for i in range(len(frame['description'])) if frame['description'].startswith('ARG', i)])
                if arg_cnt >= 2:
                    for word, tag in zip(words, frame["tags"]):
                        if tag != "O":
                            buffer.append(word)

                    frames.append(" ".join(buffer))

        except Exception as e:
            print(e)

        return frames

    def get_frames_tag(self, sample):
        frames = []
        try:
            res = self.srl_predictor.predict(
                sentence=sample
            )
            words = res["words"]
            for frame in res["verbs"]:
                arg_cnt = 0
                buffer = []

                for word, tag in zip(words, frame["tags"]):
                    if tag != "O":
                        buffer.append(word)
                    if 'ARG' in tag:
                        arg_cnt += 1

                if arg_cnt >= 2:
                    buffer.append(".")
                    frames.append(" ".join(buffer))

        except Exception as e:
            print(e)

        return frames


if __name__ == '__main__':
    predictor = SRLPredictor()
    r = predictor.get_frames("India has sent 100,000 doses of COVID-19 vaccines to Barbados, and they arrived earlier today. This is a significant and meaningful gesture, and the people of Barbados are grateful to the Prime Minister of India, Mr. Modi, for his prompt and generous decision to send these vaccines. Thank you, Mr. Modi.")
