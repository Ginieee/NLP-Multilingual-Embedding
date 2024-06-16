import numpy as np


def read_and_process_file(file_path):
    word_pairs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 각 줄이 비어있지 않고 탭으로 나눠진 두 개의 값을 가져오는지 확인
            parts = line.strip().split('\t')
            if len(parts) == 2:
                eng, kor = parts
                eng = eng.replace(' ', '-')
                kor = kor.replace(' ', '-')
                word_pairs.append((eng, kor))
            else:
                print(f"Skipping invalid line: {line.strip()}")
    return word_pairs


def save_processed_pairs(word_pairs, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for eng, kor in word_pairs:
            file.write(f"{eng}\t{kor}\n")


file_path = 'data/doosan-it.txt'
output_file_path = 'data/processed-doosan-it.txt'
word_pairs = read_and_process_file(file_path)
save_processed_pairs(word_pairs, output_file_path)