import numpy as np
import pandas as pd
from wordSim import word_pairs
from wordSim import word_pairs_ko
from readVectors import src_ko_embeddings, src_ko_id2word, src_ko_word2id, src_en_embeddings, src_en_id2word, src_en_word2id, tgt_ko_embeddings, tgt_ko_id2word, tgt_ko_word2id, tgt_en_embeddings, tgt_en_id2word, tgt_en_word2id



def calculate_similarity(word1, word2, srcEmb, srcId2word):
    src_word2id = {v: k for k, v in srcId2word.items()}

    if word1 not in src_word2id or word2 not in src_word2id:
        raise ValueError("One or both words are not in the dictionary.")

    word1_emb = srcEmb[src_word2id[word1]]
    word2_emb = srcEmb[src_word2id[word2]]

    word1_emb_norm = word1_emb / np.linalg.norm(word1_emb)
    word2_emb_norm = word2_emb / np.linalg.norm(word2_emb)

    similarity = np.dot(word1_emb_norm, word2_emb_norm)
    return round(similarity * 10, 4)

def get_similarity(words, srcEmb, srcId2word, name, path):
    records = []
    for word1, word2, wordSimScore in words:
        try:
            modelScore = calculate_similarity(word1, word2, srcEmb, srcId2word)
        except (ValueError, KeyError) as e:
            modelScore = None
            print(e)
        records.append({'word1': word1, 'word2': word2, 'wordSimScore': wordSimScore, 'modelScore': modelScore})

    df = pd.DataFrame(records)
    full_path = f"{path}/{name}.csv"
    df.to_csv(full_path, index=False)
    print(f"DataFrame saved to {full_path}")


get_similarity(word_pairs_ko, src_ko_embeddings, src_ko_id2word, "wordSim_ko_src.Sim", "data/score")
get_similarity(word_pairs_ko, tgt_ko_embeddings, tgt_ko_id2word, "wordSim_ko_tgt.Sim", "data/score")

get_similarity(word_pairs, src_en_embeddings, src_en_id2word, "wordSim_en_src.Sim", "data/score")
get_similarity(word_pairs, tgt_en_embeddings, tgt_en_id2word, "wordSim_en_tgt.Sim", "data/score")
