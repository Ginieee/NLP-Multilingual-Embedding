import io
import numpy as np
from wordSim import word_pairs
from sklearn.metrics.pairwise import cosine_similarity

src_path = 'MUSE/dumped/debug/koen7/vectors-ko.txt'
tgt_path = 'MUSE/dumped/debug/koen7/vectors-en.txt'
nmax = 2000000


def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)


def calculate_similarity(word1, word2, embeddings, word2id):
    if word1 not in word2id or word2 not in word2id:
        raise ValueError("One or both words are not in the dictionary.")
    word1_emb = embeddings[word2id[word1]]
    word2_emb = embeddings[word2id[word2]]
    word1_emb_norm = word1_emb / np.linalg.norm(word1_emb)
    word2_emb_norm = word2_emb / np.linalg.norm(word2_emb)
    similarity = np.dot(word1_emb_norm, word2_emb_norm)
    return similarity


def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=10):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    result = []
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
        mid = (tgt_id2word[idx], scores[idx] * 10)
        result.append(mid)


src_word = '머신러닝'
get_nn(src_word, src_embeddings, src_id2word, src_embeddings, src_id2word, K=10)
get_nn(src_word, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, K=10)


def storeResult(words, src_emb, src_id2, tgt_emb, tgt_id2, K=10):
    total = {}
    totalFromSrc = {}
    for w1, w2, score in words:
        # print("Eng: %s, Kor: %s" % (eng, kor))
        res = get_nn(w1, src_emb, src_id2, tgt_emb, tgt_id2, K=K)
        totalFromSrc.update({eng: })
    print(total)
    return total

import csv

def save_output_to_csv(output, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['English Word', 'Nearest Korean Words'])
        for eng_word, nearest_kor_words in output.items():
            writer.writerow([eng_word, ', '.join(nearest_kor_words)])


outputKr = storeResult(word_pairs, tgt_embeddings, tgt_id2word, src_embeddings, src_id2word, K=10)
outputKr.values()

# Example usage
output_file_path = 'data/output/outputKrinKr_koen5.csv'
save_output_to_csv(outputKr, output_file_path)