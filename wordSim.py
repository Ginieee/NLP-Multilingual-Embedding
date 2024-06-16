import csv


def read_wordsim353(file_path):
    word_pairs = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        for row in reader:
            word1, word2, score = row
            word_pairs.append((word1.strip(), word2.strip(), float(score)))
    return word_pairs


# Paths
wordsim_file_path = 'data/wordsim353crowd.csv'
wordsim_ko_file_path = 'data/wordsim353crowd_kor.csv'
# Load data
word_pairs = read_wordsim353(wordsim_file_path)
word_pairs_ko = read_wordsim353(wordsim_ko_file_path)
# print(word_pairs_ko)