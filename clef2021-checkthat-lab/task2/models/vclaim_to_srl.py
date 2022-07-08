from glob import glob
import json
from srl_predictor import SRLPredictor
import encoders
import pickle


def load_vclaims(dir):
    vclaims_fp = glob(f'{dir}/*.json')
    vclaims_fp.sort()
    vclaims = {}
    for vclaim_fp in vclaims_fp:
        with open(vclaim_fp) as f:
            vclaim = json.load(f)
        vclaims[vclaim['vclaim_id']] = vclaim
    return vclaims


predictor = SRLPredictor()
encoder = encoders.SBERTSentenceEncoder()

vclaims_dir = '../data/subtask-2a--english/v1/train/vclaims/'
vclaims = load_vclaims(vclaims_dir)

vclaims_srl = {}
vclaims_cnt = 0
for id in vclaims:
    vclaims_cnt += 1
    if vclaims_cnt % 1000 == 0:
        print('vclaims_cnt: {}'.format(vclaims_cnt))
    vclaim = vclaims[id]['vclaim']
    vclaim_frames = []
    vclaim_frames = predictor.get_frames(vclaim)
    vclaim_frames.append(vclaim)
    vclaim_frames_encoding = []
    vclaims_srl[id] = vclaim_frames
    # for vclaim_frame in vclaim_frames:
    #     vclaim_frames_encoding.append(encoder.encode(vclaim_frame))

with open('vclaims_srl-ARG0.pkl', 'wb') as f:
    pickle.dump(vclaims_srl, f)
