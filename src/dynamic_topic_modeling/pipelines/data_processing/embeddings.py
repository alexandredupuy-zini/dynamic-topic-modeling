import requests 
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import torch 

def download_embeddings() : 
    print('Downloading data.... ~800MB')
    r=requests.get('http://nlp.stanford.edu/data/glove.6B.zip')
    print('Download done')
    zipfile = ZipFile(BytesIO(r.content))
    print('Saving file to catalog...')
    embeddings=zipfile.open('glove.6B.300d.txt').read()

    return embeddings

def get_embeddings(glove_embeddings,emb_size,vocab,fill_embeddings) : 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings = np.zeros((len(vocab), emb_size))
    words_found = 0
    count=0
    for i, word in enumerate(vocab.values()):
        if count%1000==0 :
            print('Done processing {} out of {} words'.format(count,len(vocab)))
        try: 
            embeddings[i] = glove_embeddings[word]
            words_found += 1
        except KeyError:
            if fill_embeddings==0 : 
                pass
            elif fill_embeddings=="normal" : 
                embeddings[i] = np.random.normal(scale=0.6, size=(emb_size, ))
        count+=1

    embeddings = torch.from_numpy(embeddings).to(device)
    print("Number of words found on embedding : {} out of {}".format(words_found,len(vocab)))

    return embeddings 