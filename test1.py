import pandas as pd
import numpy as np

df=pd.read_csv('processed/embeddings.csv', index_col=0)
print('end read')
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.to_csv('processed/embeddings1.csv')