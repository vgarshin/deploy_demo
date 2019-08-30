import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pyodbc
import pandas as pd 
import numpy as np
import re
import xgboost as xgb
import utils
import json
import pickle
from time import gmtime, strftime
from random import getrandbits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals import joblib
from flask import Flask, request, Response
from multiprocessing import Process

PREDICTORS = {}
METRICS = {}
CLASSES = {}
TFIDF_VECTORIZERS = {}
MODELS_PATH = './models/'
METRICS_PATH = './metrics/'
TFIDF_PATH = './tfidf/'
CLASSES_PATH = './classes/'
CORES = 10 #10 cores: 18:04-18:07-18:18 / 1 core: 18:18-18:21-18:58(not end)
EARLY_STOP_RNDS = 5
app = Flask(__name__)

if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
if not os.path.exists(METRICS_PATH):
    os.makedirs(METRICS_PATH)
if not os.path.exists(TFIDF_PATH):
    os.makedirs(TFIDF_PATH)
if not os.path.exists(CLASSES_PATH):
    os.makedirs(CLASSES_PATH)

def get_dataset(n_news):
    qry_news = 'select * from outer_data.interfax_news limit {}'.format(n_news)
    with pyodbc.connect('DSN=Impala;Database=prod_dct_sbx', autocommit=True) as conn:
        df_news = pd.read_sql(qry_news, conn)
    print('SQL query completed')
    return utils.process_dataset(df_news)
def tfidf_features(X_train, X_val, X_test):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5, token_pattern='(\S+)')
    tfidf_vectorizer.fit(X_train)
    X_train = tfidf_vectorizer.transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    return X_train, X_val, X_test, tfidf_vectorizer
def binarize(y_train, y_val, y_test):
    lb = LabelBinarizer()
    lb.fit(y_train)
    return lb.transform(y_train), lb.transform(y_val), lb.transform(y_test), lb
def get_train_test_val(df_news):
    X_train, X_test, y_train, y_test = train_test_split(df_news['content_prc'], df_news['label'], 
                                                        test_size=.2, 
                                                        stratify=df_news['label'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                      test_size=.25, 
                                                      stratify=y_train)
    print('train: ', len(X_train), ' | val: ', len(X_val), ' | test: ', len(X_test))
    X_train, X_val, X_test, tfidf_vectorizer = tfidf_features(X_train, X_val, X_test)
    tfidf_vocab = tfidf_vectorizer.vocabulary_
    tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}
    print('length of tf-idf vocab: ', len(tfidf_vocab))
    y_train, y_val, y_test, lb = binarize(y_train, y_val, y_test)
    return X_train, X_test, X_val, y_val, y_train, y_test, tfidf_vectorizer, tfidf_reversed_vocab, lb
def train_model(n_news, n_folds, model_id):
    estimators = []
    params = []
    df_news = get_dataset(n_news)
    print('dataframe processed')
    X_train, X_test, X_val, y_val, y_train, y_test, tfidf_vectorizer, tfidf_reversed_vocab, lb = get_train_test_val(df_news)
    print('classes labels:', lb.classes_)
    print('train-test-val done')
    print('start training model {}...'.format(model_id))
    for i in range(y_train.shape[1]):
        fit_params = {
            'early_stopping_rounds': EARLY_STOP_RNDS,
            'eval_metric': 'auc', 
            'eval_set': [(X_val, y_val[:, i])],
            'verbose': [0]
        }
        param_grid = {
            'max_depth': [1, 3, 5],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 150, 300],
            'n_jobs': [CORES],
            'scale_pos_weight': [y_train[y_train[:, i] == 0, i].shape[0] / y_train[y_train[:, i] == 1, i].shape[0]],
            'random_state': [1980]
            }
        est = GridSearchCV(
            xgb.XGBClassifier(),
            param_grid,
            fit_params=fit_params,
            scoring='roc_auc',
            cv=StratifiedKFold(n_splits=n_folds),
            verbose=0
        )
        est.fit(X_train, y_train[:, i])
        estimators.append(est.best_estimator_)
        params.append(est.best_params_)
        print('model for label {} done'.format(i))
    print('tests model {} starting...'.format(model_id))
    y_preds_train = []
    y_preds_test = []
    for est in estimators:
        y_preds_train.append(list(est.predict_proba(X_train)[:, 1]))
        y_preds_test.append(list(est.predict_proba(X_test)[:, 1]))
    micro_roc_auc_train = roc_auc_score(y_train, np.array(y_preds_train).T, average='weighted')
    micro_roc_auc_test = roc_auc_score(y_test, np.array(y_preds_test).T, average='weighted')
    print('  train quality:')
    print('    roc_auc', micro_roc_auc_train)
    print('    gini', 2 * micro_roc_auc_train - 1)
    print('  test quality:')
    print('    roc_auc', micro_roc_auc_test)
    print('    gini', 2 * micro_roc_auc_test - 1)
    metrics = [
        {'train ROC-AUC': micro_roc_auc_train}, 
        {'test ROC-AUC': micro_roc_auc_test}
    ]
    logs_dict = {model_id: metrics}
    log_file_path = '{}{}.txt'.format(METRICS_PATH, model_id)
    with open(log_file_path, 'w') as file:
        json.dump(logs_dict, file)
    print('metrics saved')
    models_dict = {model_id: estimators}
    models_file_path = '{}{}.pkl'.format(MODELS_PATH, model_id)
    with open(models_file_path, 'wb') as file:
        pickle.dump(models_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('model saved')
    tfidf_dict = {model_id: tfidf_vectorizer}
    tfidf_file_path = '{}{}.pkl'.format(TFIDF_PATH, model_id)
    with open(tfidf_file_path, 'wb') as file:
        pickle.dump(tfidf_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('tfidf vectorizer saved')
    classes_dict = {model_id: lb.classes_}
    classes_file_path = '{}{}.pkl'.format(CLASSES_PATH, model_id)
    with open(classes_file_path, 'wb') as file:
        pickle.dump(classes_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('classes saved')
    print('...train for model {} completed'.format(model_id))
def theme_validate_train():
    errors = []
    json = request.get_json()
    if json is None:
        errors.append('no JSON sent, check Content-Type header')
        return (None, errors)
    for field_name in ['n_news', 'n_folds']:
        if type(json.get(field_name)) is not int:
            errors.append('field {} is missing or is not an integer'.format(field_name))
    print('got json: ', json, ' | errors: ', errors)
    return (json, errors)
def theme_validate_predict():
    errors = []
    json = request.get_json()
    if json is None:
        errors.append('no JSON sent, check Content-Type header')
        return (None, errors)
    for field_name in ['model_id', 'news_text']:
        if type(json.get(field_name)) is not str:
            errors.append('field {} is missing or is not an string'.format(field_name))
    print('got json: ', json, ' | errors: ', errors)
    return (json, errors)
def resp(code, data):
    return Response(status=code, mimetype='application/json', response=json.dumps(data))

@app.route('/train', methods=['POST'])
def get_trained_model():
    (json, errors) = theme_validate_train()
    if errors:
        return resp(400, {'errors': errors})
    n_news = json['n_news']
    n_folds = json['n_folds']
    model_id = '{}_{}'.format(strftime('%Y%m%d_%H%M%S', gmtime()), hex(getrandbits(64)))
    process = Process(
        target=train_model, 
        args=(n_news, n_folds, model_id)
    )
    process.start()
    return resp(200, {'training model' : model_id, 'news total': n_news, 'folds': n_folds})
@app.route('/load', methods=['GET'])
def get_models_loaded():
    errors = []
    try:
        for file_name in [x for x in os.listdir(MODELS_PATH) if not x.startswith('.')]:
            with open('{}{}'.format(MODELS_PATH, file_name), 'rb') as file:
                PREDICTORS.update(pickle.load(file))
    except:
        errors.append('could not load models, not ready or do not exist')
    try:
        for file_name in [x for x in os.listdir(METRICS_PATH) if not x.startswith('.')]:
            with open('{}{}'.format(METRICS_PATH, file_name), 'r') as file:
                METRICS.update(json.load(file))
    except:
        errors.append('could not load metrics, not ready or do not exist')
    try:
        for file_name in [x for x in os.listdir(TFIDF_PATH) if not x.startswith('.')]:
            with open('{}{}'.format(TFIDF_PATH, file_name), 'rb') as file:
                TFIDF_VECTORIZERS.update(pickle.load(file))
    except:
        errors.append('could not load tfidf vectorizer, not ready or do not exist')
    try:
        for file_name in [x for x in os.listdir(CLASSES_PATH) if not x.startswith('.')]:
            with open('{}{}'.format(CLASSES_PATH, file_name), 'rb') as file:
                CLASSES.update(pickle.load(file))
    except:
        errors.append('could not load classes, not ready or do not exist')
    if errors:
        return resp(400, {'errors': errors})
    else:
        return resp(200, {'models loaded': list(PREDICTORS.keys())})
@app.route('/models', methods=['GET'])
def get_models_metrics():
    return resp(200, METRICS)
@app.route('/predict', methods=['POST'])
def get_predict():
    (json, errors) = theme_validate_predict()
    if errors:
        return resp(400, {'errors': errors})
    model_id = json['model_id']
    news_text = json['news_text']
    preds = {}
    try:
        tfidf_vectorizer = TFIDF_VECTORIZERS[model_id]
        estimators = PREDICTORS[model_id]
        labels_classes = CLASSES[model_id]
    except:
        errors = 'could not load environment for prediction'
        return resp(400, {'errors': errors})
    print('env for prediction done')      
    X_test = utils.preprocessing(news_text, [])
    X_test = tfidf_vectorizer.transform([X_test])
    print('prediction text processed')
    for label_class, est in zip(labels_classes, estimators):
        proba = est.predict_proba(X_test)[:, 1]
        preds.update({str(label_class): proba.tolist()})
    return resp(200, {'model' : model_id, 'classes probabilities': preds})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True, use_reloader=False)