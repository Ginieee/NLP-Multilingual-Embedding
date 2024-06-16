import io
import numpy as np
from processFile import read_and_process_file
from wordSim import word_pairs

src_path = 'MUSE/dumped/debug/enko5/vectors-en.txt'
tgt_path = 'MUSE/dumped/debug/enko5/vectors-ko.txt'
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


def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=10):
    word2id = {v: k for k, v in src_id2word.items()}
    if word not in word2id:
        print(f"Word '{word}' not found in source embeddings.")
        return []
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    mid = []
    for i, idx in enumerate(k_best):
        mid.append(tgt_id2word[idx])
    return mid

src_word = 'kakaotalk'
get_nn(src_word, src_embeddings, src_id2word, src_embeddings, src_id2word, K=10)
get_nn(src_word, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, K=10)

processed_file_path = 'data/processed-doosan-it.txt'

def storeResult(word_pairs, src_emb, src_id2word, tgt_emb, tgt_id2word, K=10):
    totalResult = {}
    for eng, kor, score in word_pairs:
        try:
            result = get_nn(kor, src_emb, src_id2word, tgt_emb, tgt_id2word, K=K)
            totalResult[eng] = result
        except KeyError as e:
            print(f"KeyError for word pair ({eng}, {kor}): {e}")
            continue
    print(totalResult)
    return totalResult

import csv

def save_output_to_csv(output, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['English Word', 'Nearest Korean Words'])
        for eng_word, nearest_kor_words in output.items():
            writer.writerow([eng_word, ', '.join(nearest_kor_words)])


outputKr = storeResult(word_pairs, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, K=10)
outputKr.values()

# Example usage
output_file_path = 'data/output/outputKrinKr_enko5.csv'
save_output_to_csv(outputKr, output_file_path)
#
#
# def evaluate_accuracy(word_pairs, en_vectors, ko_vectors):
#     correct_count = 0
#     total_count = len(word_pairs)
#     for eng, kor in word_pairs:
#         print(f'{eng} - {kor}')
#         if eng in en_vectors and kor in ko_vectors:
#             neighbors = get_nn(eng, en_vectors, src_id2word, ko_vectors, K=10)
#             print(f"{eng} Neighbors: {neighbors}")
#             if kor in neighbors:
#                 correct_count += 1
#     return correct_count / total_count


# accuracy = evaluate_accuracy(word_pairs, src_embeddings, tgt_embeddings)
# print(f"Accuracy: {accuracy * 100:.2f}%")
