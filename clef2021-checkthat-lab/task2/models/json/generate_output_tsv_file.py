import json
import pickle


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"


with open('sbert-srl-clef2021-2a-train.pkl', 'rb') as f:
    retrieval_results = pickle.load(f)

with open('claims-id-to-idx.json', 'r') as f:
    claims_id_to_idx = json.load(f)

with open('iclaims-id-train.pkl', 'rb') as f:
    iclaims_id = pickle.load(f)


f = open('SBERT-SRL-FCCKB-0729.tsv', 'w')
iclaim_id_idx = 0
for retrieval_result in retrieval_results:
    iclaim_id = iclaims_id[iclaim_id_idx]
    score = 0.6
    for vclaim_idx in retrieval_result:
        vclaim_id = get_key(vclaim_idx, claims_id_to_idx)
        f.write('{}\t0\t{}\t1\t{}\tSBERT-SRL-FCCKB\n'.format(iclaim_id, vclaim_id, score))
        score -= 0.03
    iclaim_id_idx += 1

f.close()
