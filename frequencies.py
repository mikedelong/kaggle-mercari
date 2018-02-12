import logging
import re
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

start_time = time.time()

# set up logging
formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

stopwords = {'&', '[rm]', 'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any',
             'are', 'aren', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
             'brand new', 'but', 'by', 'can', 'couldn', 'd', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don',
             'down', 'during', 'each', 'few', 'for', 'free ship.*?', 'from', 'further', 'had', 'hadn', 'has', 'hasn',
             'have', 'haven', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i',
             'if', 'in', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', 'more',
             'most', 'mustn', 'my', 'myself', 'needn', 'new', 'no', 'no description yet', 'nor', 'not', 'now', 'o',
             'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
             'price firm', 're', 'rm', 's', 'same', 'shan', 'she', 'should', 'shouldn', 'so', 'some', 'such', 't',
             'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this',
             'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', 'we', 'were',
             'weren', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', 'wouldn',
             'y', 'you', 'your', 'yours', 'yourself', 'yourselves'}

logger.debug(sorted(stopwords))
pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')

# todo move this to a settings file
input_folder = './input/'
train_file = 'train.tsv'
full_train_file = input_folder + train_file
converters = {'item_description': lambda x: pattern.sub('', x.lower()), 'name': lambda x: pattern.sub('', x.lower())}
logger.debug('loading training data from %s' % full_train_file)
encoding = 'utf-8'
train = pd.read_csv(full_train_file, sep="\t", encoding=encoding, converters=converters)
logger.debug('training data load complete.')
logger.debug('training data has %d columns and %d rows' % train.shape)
logger.debug('training data has columns %s' % train.columns)

if False:
    min_df = 3000
    name_vectorizer = TfidfVectorizer(min_df=min_df)
    name_vectorizer_model = name_vectorizer.fit(train['name'])

max_df = 20000
description_vectorizer = TfidfVectorizer(max_features=max_df, ngram_range=(1, 3))
description_model = description_vectorizer.fit(train['item_description'])

description_vocabulary = description_model.vocabulary_
description_stopwords = description_model.stop_words_
logger.debug('with max_df = %d we have %d words in the vocabulary and %d stopwords.' %
             (max_df, len(description_vocabulary), len(description_stopwords)))

idf = description_model.idf_

logger.debug('%.3f %.3f' % (idf.min(), idf.max()))
logger.debug(pd.Series(idf).describe())

sorted_terms = [item for item in sorted(zip(idf, description_vocabulary.keys()), reverse=True)]
for item in sorted_terms[:10]:
    logger.debug(item)

finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
