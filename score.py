import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('data/score/wordSim_ko_src.Sim.csv')
df_en = pd.read_csv('data/score/wordSim_en_src.Sim.csv')

# wordSimScore와 modelScore의 차이 계산
df['difference'] = abs(df['wordSimScore'] - df['modelScore'])
df_en['difference'] = abs(df_en['wordSimScore'] - df_en['modelScore'])

# 각 행의 성능 계산 (10점 만점 기준)
df['performance'] = 100 - (df['difference'] / 10 * 100)
df_en['performance'] = 100 - (df_en['difference'] / 10 * 100)

# 전체 성능 계산 (각 행의 성능의 평균)
overall_performance_percentage = df['performance'].mean()
overall_performance_percentage_en = df_en['performance'].mean()

# 결과 출력
print(f"Overall performance KO: {overall_performance_percentage:.2f}%")
print(f"Overall performance EN: {overall_performance_percentage_en:.2f}%")
