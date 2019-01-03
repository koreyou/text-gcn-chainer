import re

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_en = set(stopwords.words('english'))


def normalize_text(string):
    """ Text normalization from
    https://github.com/yoonkim/CNN_sentence/blob/23e0e1f735570/process_data.py
    as specified in Yao's paper.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def tokenize(text):
    return [t for t in normalize_text(text).split() if t not in stopwords_en]
