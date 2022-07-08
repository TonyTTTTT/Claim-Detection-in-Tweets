import pandas as pd
import json
from glob import glob
import pickle


train_dev_iclaim = '../data/subtask-2a--english/v1/train/tweets-train-dev.tsv'
test_iclaim = '../test-gold/subtask-2a--english/subtask-2a--english/tweets-test.tsv'
iclaims_path = test_iclaim
vclaims_dir = '../data/subtask-2a--english/v1/train/vclaims/'

qrels_train_path = '../data/subtask-2a--english/v1/train/qrels-train.tsv'
qrels_dev_path = '../data/subtask-2a--english/v1/train/qrels-dev.tsv'
qrels_test_path = '../test-gold/subtask-2a--english/subtask-2a--english/qrels-test.tsv'


class DataLoader:
    def __init__(self):
        self.iclaims_df = pd.read_csv(iclaims_path, sep='\t', names=['id', 'text'])
        self.iclaims = self.iclaims_to_dict(self.iclaims_df)
        self.qrels_train = pd.read_csv(qrels_train_path, sep='\t', names=['iclaim_id', 'nonsense', 'vclaim_id', 'relevance'])
        self.qrels_dev = pd.read_csv(qrels_dev_path, sep='\t', names=['iclaim_id', 'nonsense', 'vclaim_id', 'relevance'])
        self.qrels_test = pd.read_csv(qrels_test_path, sep='\t', names=['iclaim_id', 'nonsense', 'vclaim_id', 'relevance'])
        self.vclaims = self.load_vclaims(vclaims_dir)

    @staticmethod
    def load_vclaims(dir):
        vclaims_fp = glob(f'{dir}/*.json')
        vclaims_fp.sort()
        vclaims = {}
        for vclaim_fp in vclaims_fp:
            with open(vclaim_fp) as f:
                vclaim = json.load(f)
            vclaims[vclaim['vclaim_id']] = [vclaim['vclaim']]
        return vclaims

    # for training supervised ML usage
    # def df_to_lists(self, df):
    #     iclaim_ids = []
    #     relevances
    #     for row in df.iterrows():
    #         iclaim_id = row[1][0]
    #         vclaim_id = row[1][2]
    #         relevance = row[1][3]
    #         iclaim = self.iclaims.loc[self.iclaims['id'] == iclaim_id]['text'].item()

    @staticmethod
    def iclaims_to_dict(iclaims):
        iclaims_dict = {}
        iclaims.drop(index=iclaims.index[0], axis=0, inplace=True)  # drop first row(column name)
        for row in iclaims.iterrows():
            iclaims_dict[row[1]['id']] = row[1]['text']

        return iclaims_dict

    def get_iclaims(self):
        return self.iclaims

    def get_vclaims(self):
        return self.vclaims

    def get_vclaims_srl(self):
        vclaims_srl_f = open('vclaims_srl.pkl', 'rb')
        vclaims_srl = pickle.load(vclaims_srl_f)
        vclaims_srl_f.close()

        return vclaims_srl

    def get_qrels_train(self):
        return self.qrels_train

    def get_qrels_dev(self):
        return self.qrels_dev

    def get_qrels_test(self):
        return self.qrels_test



if __name__ == '__main__':
    data_loader = DataLoader()
    iclaims = data_loader.get_iclaims()
    vclaims = data_loader.get_vclaims()
