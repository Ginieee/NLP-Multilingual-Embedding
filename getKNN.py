from readVectors import src_ko_embeddings, src_ko_id2word, src_ko_word2id, src_en_embeddings, src_en_id2word, src_en_word2id, tgt_ko_embeddings, tgt_ko_id2word, tgt_ko_word2id, tgt_en_embeddings, tgt_en_id2word, tgt_en_word2id
import io
import numpy as np
import pandas as pd

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    if word not in word2id:
        return []
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    return [tgt_id2word[idx] for idx in k_best]
    # mid = []
    # for i, idx in enumerate(k_best):
    #     print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
    #     mid.append(tgt_id2word[idx])
    #
    # return mid

# processed-doosan-it.txt 파일에서 영단어와 한국어 단어 쌍을 읽어오기
word_pairs = []
with open('data/processed-doosan-it.txt', 'r', encoding='utf-8') as f:
    for line in f:
        en_word, ko_word = line.strip().split('\t')
        word_pairs.append((en_word, ko_word))

# 결과 저장 리스트
results = []
naCnt = 0
# 각 영단어에 대해 get_nn 호출
for en_word, ko_word in word_pairs:
    # 영어 단어의 KNN 얻기
    knn_ko_words = get_nn(en_word, src_en_embeddings, src_en_id2word, tgt_ko_embeddings, tgt_ko_id2word, K=10)
    if not knn_ko_words:
        results.append((en_word, ko_word, 'N/A', False))
        naCnt += 1
        continue

    # KNN 결과에서 한국어 단어와의 일치 여부 확인
    match_found = False
    normalized_ko_word = ko_word.replace(" ", "")
    for knn_word in knn_ko_words:
        normalized_knn_word = knn_word.replace(" ", "")
        if normalized_knn_word == normalized_ko_word or normalized_ko_word in normalized_knn_word:
            match_found = True
            break

    # 결과 리스트에 저장
    results.append((en_word, ko_word, ', '.join(knn_ko_words), match_found))


# 결과를 데이터프레임으로 변환
results_df = pd.DataFrame(results, columns=['English Word', 'Korean Word', 'KNN Results', 'Match Found'])

# 결과를 CSV 파일로 저장
results_df.to_csv('data/output/knn_results.csv', index=False)

print(results_df['Match Found'].value_counts())
# True 비율 계산
true_ratio = results_df['Match Found'].mean() * 100
print(f"True ratio: {true_ratio:.2f}%")
print(naCnt)

print(results_df.head(5))


# printing nearest neighbors in the source space
src_word = 'machine-learning'
src_en_words = get_nn(src_word, src_en_embeddings, src_en_id2word, src_en_embeddings, src_en_id2word, K=10)
tgt_ko_words = get_nn(src_word, src_en_embeddings, src_en_id2word, tgt_ko_embeddings, tgt_ko_id2word, K=10)

src_word = '머신러닝'
src_ko_words = get_nn(src_word, src_ko_embeddings, src_ko_id2word, src_ko_embeddings, src_ko_id2word, K=10)
tgt_en_words = get_nn(src_word, src_ko_embeddings, src_ko_id2word, tgt_en_embeddings, tgt_en_id2word, K=10)