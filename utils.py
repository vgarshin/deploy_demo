import os
import pyodbc
import pandas as pd 
import numpy as np
import pymorphy2
import re
from bs4 import BeautifulSoup
from pymorphy2 import tokenizers

morph = pymorphy2.MorphAnalyzer()

def get_features(column):
    output1, output2, output3 = [], [], []
    for el in column:
        try:
            soup = BeautifulSoup(el, 'lxml')
            output1.append(re.sub('<.*?>|\t', '', el))
            output2.append({key: len(soup.findAll('span', {'data-entity-tonality': key})) for key in [-1, 0, 1]})
            output3.append([elm.text for elm in soup.findAll('span', {'class': 'company'})])
        except:
            print(el)
    return output1, output2, output3
def split_word(word):
    for i in range(1, len(word)):
        head, tail = word[:i], word[i:]
        if morph.word_is_known(head) and morph.word_is_known(tail):
            return head, tail
    return word
def preprocessing(sentence, clist):
    s = re.sub('[^а-яА-Яa-zA-Z]+', ' ', sentence).strip().lower()
    s = re.sub('ё', 'е', s)
    result = []
    for word in tokenizers.simple_word_tokenize(s):
        if word not in clist:
            if not morph.word_is_known(word):
                new_words = split_word(word)
            else:
                new_words = word,
            for new_word in new_words:
                parse = morph.parse(new_word)[0]
                pos = parse.tag.POS
                if pos is not None and pos not in ['NPRO', 'PREP', 'CONJ', 'PRCL', 'INTJ']:
                    result.append(parse.normal_form)
        else:
            result.append(word)
    return ' '.join(result)
def preprocessing_cps(clist):
    result = []
    for c in clist:
        s = re.sub('[^а-яА-Яa-zA-Z]+', ' ', c).strip().lower()
        s = re.sub('ё', 'е', s)
        result.append(s)
    return list(set(result))
def process_dataset(df_news):
    cols_to_drop = ['author', 'dedupclusterid', 'fulltextunavailable', 'loadingdate', 'sourceid', 'sourcename', 'url']
    df_news.drop(cols_to_drop, axis=1, inplace=True)
    print('trash columns deleted...')
    df_news = df_news.fillna('')
    df_news['title_txt'], df_news['title_tns'], df_news['title_cps'] = get_features(df_news['title'])
    cols_to_drop = ['title']
    df_news.drop(cols_to_drop, axis=1, inplace=True)
    print('title features done...')
    df_news['content_txt'], df_news['content_tns'], df_news['content_cps'] = get_features(df_news['content'])
    cols_to_drop = ['content']
    df_news.drop(cols_to_drop, axis=1, inplace=True)
    print('content features done...')
    df_news['content_cps_prc'] = df_news['content_cps'].apply(preprocessing_cps)
    cols_to_drop = ['content_cps']
    df_news.drop(cols_to_drop, axis=1, inplace=True)
    print('content companies processed...')
    df_news['title_cps_prc'] = df_news['title_cps'].apply(preprocessing_cps)
    cols_to_drop = ['title_cps']
    df_news.drop(cols_to_drop, axis=1, inplace=True)
    print('title companies processed...')
    df_news['content_prc'] = [preprocessing(row['content_txt'], row['content_cps_prc']) for _, row in df_news.iterrows()]
    cols_to_drop = ['content_txt', 'content_cps_prc']
    df_news.drop(cols_to_drop, axis=1, inplace=True)
    print('content processed...')
    df_news['title_prc'] = [preprocessing(row['title_txt'], row['title_cps_prc']) for _, row in df_news.iterrows()]
    cols_to_drop = ['title_txt', 'title_cps_prc']
    df_news.drop(cols_to_drop, axis=1, inplace=True)
    print('title processed...')
    df_news['label'] = df_news['content_tns'].apply(lambda x: max(x, key=x.get))
    print(print(df_news['label'].value_counts()))
    cols_to_drop = ['title_tns', 'content_tns']
    df_news.drop(cols_to_drop, axis=1, inplace=True)
    print('tonality label done...')
    return df_news