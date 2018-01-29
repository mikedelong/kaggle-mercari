# https://www.kaggle.com/kenwat/single-lgbm-lb-0-44679-run-time-1687sec

import logging
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

start_time = time.time()


def split_categories(arg_text):
    try:
        category_names = arg_text.split("/")
        if len(category_names) >= 3:
            return category_names[0], category_names[1], category_names[2]
        if len(category_names) == 2:
            return category_names[0], category_names[1], 'missing'
        if len(category_names) == 1:
            return category_names[0], 'missing', 'missing'
    except:
        return ("missing", "missing", "missing")


# set up logging
formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

# todo sort these and/or move them to an external data file
stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
             'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
             'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', '&', 'brand new', 'new', '[rm]',
             'free ship.*?', 'rm', 'price firm', 'no description yet'}

pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')

input_folder = './input/'
train_file = 'train.tsv'
full_train_file = input_folder + train_file
converters = {'item_description': lambda x: pattern.sub('', x.lower()), 'name': lambda x: pattern.sub('', x.lower())}
logger.debug('loading training data from %s' % full_train_file)
encoding = 'utf-8'
train = pd.read_csv(full_train_file, sep="\t", encoding=encoding, converters=converters)
logger.debug('training data load complete.')
test_file = 'test.tsv'
full_test_file = input_folder + test_file
logger.debug('loading test data from %s' % full_test_file)
test = pd.read_csv(full_test_file, sep="\t", encoding=encoding, converters=converters)
logger.debug('test data load complete.')

train_label = np.log1p(train['price'])
train_texts = train['name'].tolist()
test_texts = test['name'].tolist()

# replace missing words
train['category_name'].fillna('other', inplace=True)
test['category_name'].fillna('other', inplace=True)

train['brand_name'].fillna('missing', inplace=True)
test['brand_name'].fillna('missing', inplace=True)

test['item_description'].fillna('none', inplace=True)
train['item_description'].fillna('none', inplace=True)

test['nm_word_len'] = list(map(lambda x: len(x.split()), test_texts))
train['nm_word_len'] = list(map(lambda x: len(x.split()), train_texts))
test['desc_word_len'] = list(map(lambda x: len(x.split()), test['item_description'].tolist()))
train['desc_word_len'] = list(map(lambda x: len(x.split()), train['item_description'].tolist()))
test['nm_len'] = list(map(lambda x: len(x), test_texts))
train['nm_len'] = list(map(lambda x: len(x), train_texts))
test['desc_len'] = list(map(lambda x: len(x), test['item_description'].tolist()))
train['desc_len'] = list(map(lambda x: len(x), train['item_description'].tolist()))
test_id = test['test_id']

nrow_train = train.shape[0]

train['subcat_0'], train['subcat_1'], train['subcat_2'] = zip(
    *train['category_name'].apply(lambda x: split_categories(x)))
test['subcat_0'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_categories(x)))

NAME_MIN_DF = 30
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name_mix = count.fit_transform(train['name'].append(test['name']))
X_name = X_name_mix[:nrow_train]
X_t_name = X_name_mix[nrow_train:]

finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
