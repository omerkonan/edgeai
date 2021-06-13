from tensorflow.python.util.tf_inspect import _convert_maybe_argspec_to_fullargspec
from feature_extraction import *
import pandas as pd

def Normalization(df_feature):
    
     return MinMaxScaler().fit_transform(df_feature.drop('label', axis=1))


headers = ['label','axis1','axis2','axis3'] #first column has to be label column
label_list = ['Still','Lifting','Falling','Shaking']
#datapath = "../data/records.json"
datapath = "../data/dataset2.csv"

feature_method_name = "SMA"
windows_length = 15
normalization =  True
standardization = False
data = Data(datapath, headers, label_list, (feature_method_name, windows_length), normalization, standardization)
data.processData.to_csv("../data/dataset_SMA_15.csv",index = False)
#data = pd.read_csv('../data/dataset.txt', sep=",", header = None, engine='c', error_bad_lines=False, warn_bad_lines=False)
#data.columns = ['label','axis1','axis2','axis3']
#print("data:", data.processed_data)
print("data\n",data.processData)
#print(SMA(data,20))
print("--------------------------------------------------")
#print(data.drop('label', axis=1))

#data2 = Normalization(data.df)

#print("norm",type(data2))


"""
df_SMA = CMV(data,20)


for col in data.columns:
    print(type(col))
    print("------------------------------------------")
    print("str", str(col))
    print(data[str(col)])
    if col != "label":

print(df_SMA)



precision recall, f1 score
current epoch
preprocess
normalization
standartization
"""
