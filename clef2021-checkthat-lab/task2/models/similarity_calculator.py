import os.path

from srl_predictor import SRLPredictor
import encoders
from numpy.linalg import norm
import numpy as np
import pickle
from data_loader import DataLoader


class SimilarityCalculator:
    def __init__(self, tsv_file_name):
        self.predictor = SRLPredictor()
        self.encoder = encoders.SBERTSentenceEncoder()
        self.tsv_file_name = tsv_file_name

    def pairwise_calculate(self, iclaim_frames_encoding, vclaim_frames_encoding):
        # Pairwise calculate the similarity between each frames of the given iclam and each frames of the given vclaim
        #
        # Then choose the maximum as the similarity between given iclaim and given vclaim
        sim_max = 0
        i_max = -1
        j_max = -1
        for i in range(0, len(iclaim_frames_encoding)):
            for j in range(0, len(vclaim_frames_encoding)):
                # sim = np.dot(iclaim_frames_encoding[i], vclaim_frames_encoding[j]) / (
                #         norm(iclaim_frames_encoding[i]) * norm(vclaim_frames_encoding[j]))
                sim = np.dot(iclaim_frames_encoding[i], vclaim_frames_encoding[j])
                # print('sim: {}'.format(sim))
                if sim > sim_max:
                    sim_max = sim
                    i_max = i
                    j_max = j
        # print('sim_max: {}, i_max: {}, j_max: {}'.format(sim_max, i_max, j_max))

        return sim_max

    def sim_calculate(self, iclaims, vclaims):
        iclaim_cnt = 0
        for iclaim_id in iclaims.keys():  # iterate through each input tweets(claims)
            iclaim = iclaims[iclaim_id]
            iclaim_frames = self.predictor.get_frames(iclaim)
            iclaim_frames.append(iclaim)
            iclaim_frames_encoding = []
            for iclaim_frame in iclaim_frames:
                iclaim_frames_encoding.append(self.encoder.encode(iclaim_frame))

            sim_list = []
            for vclaim_id in vclaims.keys():  # iterate through each verified claims
                vclaim_frames = vclaims[vclaim_id]
                vclaim_frames_encoding = []
                for vclaim_frame in vclaim_frames:
                    vclaim_frames_encoding.append(self.encoder.encode(vclaim_frame))

                sim_max = self.pairwise_calculate(iclaim_frames_encoding, vclaim_frames_encoding)
                sim_list.append([vclaim_id, sim_max])
            sim_list = sorted(sim_list, key=lambda item: item[1], reverse=True)

            # Save the ranking result into tsv file
            if os.path.exists(self.tsv_file_name):
                os.remove(self.tsv_file_name)
            with open(self.tsv_file_name, 'a') as f:
                for i in range(0, 20):
                    f.write('{}\t0\t{}\t1\t{}\tSBERT\n'.format(iclaim_id, sim_list[i][0], sim_list[i][1]))
            iclaim_cnt += 1
            print('{}th iclaim finish!'.format(iclaim_cnt))


if __name__ == '__main__':
    train_dev_iclaim = '../data/subtask-2a--english/v1/train/tweets-train-dev.tsv'
    test_iclaim = '../test-gold/subtask-2a--english/subtask-2a--english/tweets-test.tsv'
    data_loader = DataLoader(train_dev_iclaim)
    iclaims = data_loader.get_iclaims()
    vclaims = data_loader.get_vclaims()
    similarity_calculator = SimilarityCalculator('SBERT-SRL-ARG0-0727.tsv')
    similarity_calculator.sim_calculate(iclaims, vclaims)
