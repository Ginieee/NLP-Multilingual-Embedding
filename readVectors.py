import io
import numpy as np

src_ko_path = 'MUSE/dumped/debug/koen7/vectors-ko.txt'
src_en_path = 'MUSE/dumped/debug/enko7/vectors-en.txt'
tgt_en_path = 'MUSE/dumped/debug/koen7/vectors-en.txt'
tgt_ko_path = 'MUSE/dumped/debug/enko7/vectors-ko.txt'
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


src_ko_embeddings, src_ko_id2word, src_ko_word2id = load_vec(src_ko_path, nmax)
src_en_embeddings, src_en_id2word, src_en_word2id = load_vec(src_en_path, nmax)
tgt_ko_embeddings, tgt_ko_id2word, tgt_ko_word2id = load_vec(tgt_ko_path, nmax)
tgt_en_embeddings, tgt_en_id2word, tgt_en_word2id = load_vec(tgt_en_path, nmax)


# def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=10):
#     print("Nearest neighbors of \"%s\":" % word)
#     word2id = {v: k for k, v in src_id2word.items()}
#     word_emb = src_emb[word2id[word]]
#     scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
#     k_best = scores.argsort()[-K:][::-1]
#     result = []
#     for i, idx in enumerate(k_best):
#         print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
#         mid = (tgt_id2word[idx], scores[idx] * 10)
#         result.append(mid)
#
#
# src_word = '티켓'
# get_nn(src_word, src_embeddings, src_id2word, src_embeddings, src_id2word, K=10)
# get_nn(src_word, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, K=10)
