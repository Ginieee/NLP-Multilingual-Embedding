from readVectors import src_ko_embeddings, src_ko_id2word, src_ko_word2id, src_en_embeddings, src_en_id2word, src_en_word2id, tgt_ko_embeddings, tgt_ko_id2word, tgt_ko_word2id, tgt_en_embeddings, tgt_en_id2word, tgt_en_word2id
from sklearn.decomposition import PCA
import io
import numpy as np
import matplotlib.pyplot as plt
from getKNN import src_en_words, tgt_ko_words, src_ko_words, tgt_en_words
from matplotlib import rc

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

def visualization(srcWords, srcWord2id, srcEmbeddings, tgtWords, tgtWord2id, tgtEmbeddings):
    pca = PCA(n_components=2, whiten=True)
    pca.fit(np.vstack([src_en_embeddings, tgt_ko_embeddings]))
    print('Variance explained: %.2f' % pca.explained_variance_ratio_.sum())
    plot_similar_word(srcWords, srcWord2id, srcEmbeddings, tgtWords, tgtWord2id, tgtEmbeddings, pca)


def plot_similar_word(src_words, src_word2id, src_emb, tgt_words, tgt_word2id, tgt_emb, pca):

    Y = []
    word_labels = []
    for sw in src_words:
        Y.append(src_emb[src_word2id[sw]])
        word_labels.append(sw)
    for tw in tgt_words:
        Y.append(tgt_emb[tgt_word2id[tw]])
        word_labels.append(tw)

    # find tsne coords for 2 dimensions
    Y = pca.transform(Y)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display scatter plot
    plt.figure(figsize=(10, 8), dpi=80)
    plt.scatter(x_coords, y_coords, marker='x')

    for k, (label, x, y) in enumerate(zip(word_labels, x_coords, y_coords)):
        color = 'blue' if k < len(src_words) else 'red'  # src words in blue / tgt words in red
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=19,
                     color=color, weight='bold')

    plt.xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)
    plt.ylim(y_coords.min() - 0.2, y_coords.max() + 0.2)
    plt.title('Visualization of the multilingual word embedding space')

    plt.show()

# assert words in dictionaries
# for sw in src_words:
#     assert sw in src_en_word2id, '"%s" not in source dictionary' % sw
# for tw in tgt_words:
#     assert tw in tgt_ko_word2id, '"%s" not in target dictionary' % tw

visualization(src_en_words, src_en_word2id, src_en_embeddings, tgt_ko_words, tgt_ko_word2id, tgt_ko_embeddings)