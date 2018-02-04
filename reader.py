# https://www.kaggle.com/kenwat/single-lgbm-lb-0-44679-run-time-1687sec

import gc
import logging
import re
import time

import lightgbm as lgbm
import numpy as np
import pandas as pd
from scipy import sparse as ssp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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
test_file = 'test.tsv'
full_test_file = input_folder + test_file
logger.debug('loading test data from %s' % full_test_file)
test = pd.read_csv(full_test_file, sep="\t", encoding=encoding, converters=converters)
logger.debug('test data load complete.')

train_label = np.log1p(train['price'])
logger.debug('got train labels')
train_texts = train['name'].tolist()
logger.debug('got train names')
test_texts = test['name'].tolist()
logger.debug('got test names')

# replace missing words
train['category_name'].fillna('other', inplace=True)
test['category_name'].fillna('other', inplace=True)

train['brand_name'].fillna('missing', inplace=True)
test['brand_name'].fillna('missing', inplace=True)

test['item_description'].fillna('none', inplace=True)
train['item_description'].fillna('none', inplace=True)
logger.debug('filled in missing data with -none-')

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

MAX_FEATURES_ITEM_DESCRIPTION = 20000
tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION, ngram_range=(1, 3))
X_description_mix = tv.fit_transform(train['item_description'].append(test['item_description']))
X_description = X_description_mix[:nrow_train]
X_t_description = X_description_mix[nrow_train:]
logger.debug('make categorical features')

cat_features = ['subcat_2', 'subcat_1', 'subcat_0', 'brand_name', 'category_name', 'item_condition_id', 'shipping']
for feature in cat_features:
    newlist = train[feature].append(test[feature])
    label_encoder = LabelEncoder()
    label_encoder.fit(newlist)
    train[feature] = label_encoder.transform(train[feature])
    test[feature] = label_encoder.transform(test[feature])
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(train[cat_features].append(test[cat_features]))
X_cat = one_hot_encoder.transform(train[cat_features])
X_t_cat = one_hot_encoder.transform(test[cat_features])

train_feature = ['desc_word_len', 'nm_word_len', 'desc_len', 'nm_len']
train_list = [train[train_feature].values, X_description, X_name, X_cat]
test_list = [test[train_feature].values, X_t_description, X_t_name, X_t_cat]
X = ssp.hstack(train_list).tocsr()
X_test = ssp.hstack(test_list).tocsr()

logger.debug('finished feature for training')

folds_count = 4
random_state = 2  # was 128
kfold = KFold(n_splits=folds_count, shuffle=True, random_state=random_state)

learning_rate = 0.8
num_leaves = 128
min_data_in_leaf = 1000
feature_fraction = 0.5
bagging_fraction = 0.9
bagging_freq = 1000
num_boost_round = 1000
params = {
    "bagging_fraction": bagging_fraction,
    "bagging_freq": bagging_freq,
    "boosting_type": "gbdt",
    "feature_fraction": feature_fraction,
    "learning_rate": learning_rate,
    "metric": "l2_root",
    "nthread": 4,
    "num_leaves": num_leaves,
    "objective": "regression",
    # "subsample": 0.8, # was 0.9
    "verbosity": 1}

cv_pred = np.zeros(len(test_id))
kf = kfold.split(X)
for i, (train_fold, test_fold) in enumerate(kf):
    train_t0 = time.time()
    X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], train_label[train_fold], \
                                                       train_label[test_fold]
    dtrain = lgbm.Dataset(X_train, label_train)
    dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)
    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, verbose_eval=100, early_stopping_rounds=100)
    cv_pred += bst.predict(X_test, num_iteration=bst.best_iteration)
    logger.debug('training & predict time %d', time.time() - train_t0)
    gc.collect()

cv_pred /= folds_count
cv_pred = np.expm1(cv_pred)
submission = test[["test_id"]]
submission["price"] = cv_pred
submission.to_csv("./mercari_submission.csv", index=False)

finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
