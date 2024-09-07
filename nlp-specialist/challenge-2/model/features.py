import re
from sklearn.feature_extraction.text import CountVectorizer

def clean_sentence(sent):
    sent = re.sub('[?.`*^()!°¢܌Ͱ̰ߒנ~×ҡߘ:ҰߑÍ|।;!,&%\'@#$><A-Za-z0+-9=./\'""_০-৯]', '', sent)
    sent = re.sub(r'(\W)(?=\1)', '', sent)
    sent = re.sub(r'https?:\/\/.*[\r\n]*', '', sent, flags=re.MULTILINE)
    sent = re.sub(r'\<a href', ' ', sent)
    sent = re.sub(r'&amp;', '', sent) 
    sent = re.sub(r'<br />', ' ', sent)
    sent = re.sub(r'\'', ' ', sent)
    sent = re.sub(r'ߑͰߑ̰ߒנ', '', sent)
    sent = re.sub(r'ߎɰߎɰߎɍ', '', sent)
    sent = sent.strip()
    return sent

def tokenized_data(sent):
    return sent.split()

def preprocess_text(text):
    text = [w for w in text]
    text = ' '.join(text)
    return text

def extract_features(sentences):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, lowercase=False, token_pattern=u'[\S]+')
    X = vectorizer.fit_transform(sentences)
    return X, vectorizer

def sentence_to_vector_transform(sentence, vectorizer):
    vec = vectorizer.transform([sentence])
    return vec.toarray().squeeze()
