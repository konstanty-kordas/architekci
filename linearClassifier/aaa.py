import pandas as pd
df = pd.read_csv('IRIS.csv')
df.drop(df.index[0])
print(df)