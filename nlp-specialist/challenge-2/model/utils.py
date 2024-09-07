import re

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

def remove_punctuation_and_numbers(text):
    text = [w for w in text]
    text = ' '.join(text)
    return text
