import pickle
from data_loader import DataLoader
from srl_predictor import SRLPredictor
import json

train_dev_iclaim = '../data/subtask-2a--english/v1/train/tweets-train-dev.tsv'
test_iclaim = '../test-gold/subtask-2a--english/subtask-2a--english/tweets-test.tsv'
vclaims_dir = '../data/subtask-2a--english/v1/train/vclaims/'

qrels_train_path = '../data/subtask-2a--english/v1/train/qrels-train.tsv'
qrels_dev_path = '../data/subtask-2a--english/v1/train/qrels-dev.tsv'
qrels_test_path = '../test-gold/subtask-2a--english/subtask-2a--english/qrels-test.tsv'


if __name__ == '__main__':
    predictor = SRLPredictor()
    data_loader = DataLoader(test_iclaim)
    iclaims = data_loader.get_iclaims()
    vclaims = data_loader.get_vclaims()
    qrels_train = data_loader.get_qrels_train()
    qrels_dev = data_loader.get_qrels_dev()
    qrels_test = data_loader.get_qrels_test()

    iclaims_id = []
    for iclaim_id in iclaims:
        iclaims_id.append(iclaim_id)
    with open('json/iclaims_id.pkl', 'wb') as f:
        pickle.dump(iclaims_id, f)

    claims_id_to_idx = {}
    claims = []
    idx = 0
    for vclaim_id in vclaims:
        claims.append(vclaims[vclaim_id][0])
        claims_id_to_idx[vclaim_id] = idx
        idx += 1

    evidence = []
    for row in qrels_test.iterrows():
        tmp_dict = {}
        tmp_dict['evidence-id'] = row[1]['iclaim_id']
        tmp_dict['sentence'] = iclaims[tmp_dict['evidence-id']]
        tmp_dict['claim-id'] = [claims_id_to_idx[row[1]['vclaim_id']]]
        frames = predictor.get_frames(tmp_dict['sentence'])
        tmp_dict['frames'] = frames
        evidence.append(tmp_dict)

    result = {'evidence': evidence, 'claim': claims}

    with open('json/claims_id_to_idx_test.json', 'w') as f:
        json.dump(claims_id_to_idx, f)

    with open('json/clef20212a-FCCKB-format-test.json', 'w') as f:
        json.dump(result, f)
