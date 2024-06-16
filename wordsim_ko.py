import pandas as pd
from googletrans import Translator

# wordsim353crowd.csv 파일 읽기
input_file = 'data/wordsim353crowd.csv'
df = pd.read_csv(input_file, header=None, names=['word1', 'word2', 'score'])

# Google Translate API 사용 설정
translator = Translator()

# 단어 번역
def translate_word(word):
    try:
        translation = translator.translate(word, src='en', dest='ko')
        return translation.text
    except Exception as e:
        print(f"Error translating word {word}: {e}")
        return None

# 데이터프레임에 번역된 단어 추가
df['word1_kor'] = df['word1'].apply(translate_word)
df['word2_kor'] = df['word2'].apply(translate_word)

# 번역된 단어들로 새로운 데이터프레임 생성
df_kor = df[['word1_kor', 'word2_kor', 'score']].copy()
df_kor.columns = ['word1', 'word2', 'score']
# print(df_kor)

# 번역되지 않은 단어 처리
if df_kor['word1'].isnull().any() or df_kor['word2'].isnull().any():
    missing_translations = df_kor[df_kor.isnull().any(axis=1)]
    print("다음 단어들은 번역되지 않았습니다:")
    print(missing_translations)
    df_kor = df_kor.dropna()

# wordsim353crowd_kor.csv 파일로 저장
output_file = 'data/wordsim353crowd_kor.csv'
df_kor.to_csv(output_file, index=False, header=False, encoding='utf-8-sig')
print(f"번역된 데이터가 {output_file} 파일로 저장되었습니다.")
