import pandas as pd


df= pd.read_csv("./dataset.txt", engine= 'c', header=None,sep = ",",error_bad_lines=False, warn_bad_lines=True)

print(df)
print("-----------")
df.columns = ["label","axis1","axis2", "axis3"]

print(df)
print("-----------------------------")
print(df.dtypes)