import pandas as pd

def merge_csv_files(file1, file2, file3, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df_merged = pd.concat([df1, df2, df3])
    df_merged.to_csv(output_file, index=False)

merge_csv_files('creditcard1.csv', 'creditcard2.csv', 'creditcard3.csv', 'creditcard.csv')