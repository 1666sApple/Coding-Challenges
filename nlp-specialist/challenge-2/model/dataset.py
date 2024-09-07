import os
import pandas as pd

def read_file():
    classDict = {'neg': 0, 'pos': 1, 'ntr': 2}
    sentences = []
    labels = []
    scrip_dir = os.path.dirname(os.path.realpath('__file__'))
    rel_path = "bangla_product_reviews.csv"
    abs_file_path = os.path.join(scrip_dir, rel_path)
    
    data = pd.read_csv(abs_file_path)
    
    for index, row in data.iterrows():
        sentence = row['Review']
        rating = row['Rating']
        
        if rating in [1, 2]:
            label = 'neg'
        elif rating == 3:
            label = 'ntr'
        else:
            label = 'pos'
        
        labels.append(label)
        sentences.append(sentence)
    
    print('Data loading complete')
    return sentences, labels
