def preprocess_UN(df):
    df.rename({'year':'timestamp'}, axis=1, inplace=True)
    df = df[['timestamp', 'text']].copy()
    return df
