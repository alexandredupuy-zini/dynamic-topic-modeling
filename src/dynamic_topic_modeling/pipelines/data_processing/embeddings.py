import numpy as np
import torch


def get_embeddings(filepath,vocab) :

    vectors = {}
    c=0
    with open(filepath, 'rb') as f:
        
        for l in f:
            if c%1000 == 0 :
                print(c)
            line = l.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            vectors[word] = vect
            c+=1
    embeddings = np.zeros((len(vocab),300))
    for i, word in enumerate(vocab):
        try: 
            embeddings[i] = vectors[word]
            words_found += 1
        except KeyError:
            embeddings[i] = np.random.normal(scale=0.6, size=(300, ))

    embeddings = torch.from_numpy(embeddings)

    return embeddings