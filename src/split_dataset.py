import pandas as pd
from sklearn.model_selection import train_test_split

input_files = [
    'utt_48k.csv',
    'utt_24k.csv',
    'utt_16k.csv',
]
dfs = [pd.read_csv(f) for f in input_files]
df = pd.concat(dfs, ignore_index=True)
utts = df['uttID'].unique()

train_utts, temp_utts = train_test_split(
    utts,
    test_size=0.2,      
    random_state=42,
    shuffle=True
)

val_utts, test_utts = train_test_split(
    temp_utts,
    test_size=0.5,       
    random_state=42,
    shuffle=True
)

train_df = df[df['uttID'].isin(train_utts)]
val_df   = df[df['uttID'].isin(val_utts)]
test_df  = df[df['uttID'].isin(test_utts)]


train_df.to_csv('train_mos.csv', index=False)
val_df.to_csv('val_mos.csv',   index=False)
test_df.to_csv('test_mos.csv',  index=False)