import pandas as pd

input_file = 'data/score/wordSim_en_src.Sim.csv'
df = pd.read_csv(input_file)

input_file_ko = 'data/score/wordSim_ko_src.Sim.csv'
df_ko = pd.read_csv(input_file_ko)

output = 'data/output/knn_results.csv'
df_output = pd.read_csv(output)