import pandas as pd
import os


def calculate_avg_words(articles):
    total_words = 0

    for article in articles:
        tmp_words = len(article.split())
        total_words += tmp_words

    avg_words = total_words / len(articles)

    return avg_words


for dir_name in os.listdir():
    if os.path.isdir(dir_name) and 'processed' in dir_name:
        for file_name in os.listdir(dir_name):
            if 'train' not in file_name and 'dev' not in file_name and'test' not in file_name and file_name[-3:] == 'tsv':
                df = pd.read_csv(os.path.join(dir_name, file_name), sep='\t', index_col=None, quotechar='"', quoting=3)
                avg_words = calculate_avg_words(df['text'])
                print('{}\navg_words: {}\n'.format(file_name[:-4], avg_words))

