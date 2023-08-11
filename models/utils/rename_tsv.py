import os


path = '../preprocess_datasets_GPT/'

for f in os.listdir(path):
    new_name = f
    if 'train' in f:
        new_name = f.replace('_train', '')[:-4] + '_train.tsv'
    elif 'dev' in f:
        new_name = f.replace('_dev', '')[:-4] + '_dev.tsv'
    elif 'test' in f:
        new_name = f.replace('_test', '')[:-4] + '_test.tsv'

    print('{}\n{}\n'.format(f, new_name))
    os.rename(path+f, path+new_name)