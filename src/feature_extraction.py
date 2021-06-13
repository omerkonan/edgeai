import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os 

class Data:
  """ 
  Data class has clearing process for data and has spliting process as test,train, validation  according to class parameters
  
  Init values and data parameters:

  datapath: the path where the data is stored
  savepath: the path where the data want to be stored
  seperator: seperator character for the txt file
  label_list: The list that shows how many different labels
  df: Pandas dataframe for the data
  test_rate: ratio of test data to total data
  val_rate: ratio of validation data to total data
  x_train: training features data
  y_tarin: target train data(Labels)
  x_test: test features data
  y_test: target test data(Labels)
  x_val: validation features data
  y_val: target validation data(Labels)
  
  """
  
  def __init__(self, datapath, headers, label_list, feature_method=("RAW", 20),\
              normalization=False, standardization=False, test_rate=0.2, val_rate=0.2, seperator=',', savepath="./",):
    self.datapath = datapath
    self.savepath = savepath
    self.data_type = self.findDataPath()
    self.seperator = seperator
    self.headers = headers
    self.label_list = label_list
    self.df = self.readData()
    self.feature_method, self.normalization, self.standardization = feature_method, normalization, standardization
    self.df_feature,self.df_label = self.processData(self.df.drop('label', axis=1), self.feature_method, self.normalization, self.standardization)
    self.processData = pd.concat([self.df_label, self.df_feature], axis=1)
    self.test_rate = test_rate
    self.val_rate = val_rate/(1-test_rate)
    self.x_train, self.y_train, self.x_test, self.y_test,\
        self.x_val, self.y_val = self.splitData()   
  
  def findDataPath(self):
    filename = self.datapath.split("/")[-1]
    file_extansion = filename.split(".")[-1]
    return file_extansion
  def readData(self):
    '''read data from txt, csv or json file as a dataframe'''
    if (self.data_type == 'txt'):
      df = pd.read_csv(self.datapath, sep=self.seperator, header= None, engine='c', error_bad_lines=False, warn_bad_lines=False)
      df.columns = self.headers
    elif (self.data_type == 'csv'):
      df = pd.read_csv(self.datapath, engine='c',error_bad_lines=False, warn_bad_lines=False)
    elif (self.data_type == 'json'):
      df = pd.read_json(self.datapath)
    else: 
      return(print('Error: Datatype error - Wrong DataType input'))
  
    df = self.labelencode(df)

    return df

  def labelencode(self,df):
    for i in range(len(self.label_list)):
      df = df.replace({'label':self.label_list[i]},i)
    return df

  def getDataInfo(self):
    return self.df.info()

  def cleanData(self):
    return self.df.dropna(inplace=False)

  def addLabel(self, label):
    self.df['label'] = label
    return self.df
  
  
  def getColumn(self, column_name):
    return self.df[column_name]

  def getRow(self, row_number):
    return self.df.loc[:row_number]

  
  def saveData(self, saving_path):
    if os.path.exists(saving_path):
      """ Write control question for deleting file with argparse, while working on local """
      os.remove(saving_path)
      self.df.to_csv(saving_path, header=None, index=None, \
                                sep=self.seperator)
    else:
      self.df.to_csv(saving_path, header=None, index=None, \
          sep=self.seperator)
  
  def splitData(self):

    x, x_test, y, y_test = train_test_split(self.df_feature, self.df_label,\
                                            test_size=self.test_rate, train_size=(1-self.test_rate))
    x_train, x_val, y_train, y_val = train_test_split(x, y, \
                                            test_size = self.val_rate, train_size =(1-self.val_rate))

    print("test--------------------------------------------------------")
    print(x_train, y_train, x_test, y_test, x_val, y_val)
    return x_train, y_train, x_test, y_test, x_val, y_val

  def convertConv1D(self):
    """
    Keras conv1d layer input shape should be 3d. This function adjust data dimension for con1d
    """
    x_train = np.expand_dims(self.x_train, axis=2)
    y_train = np.expand_dims(self.y_train, axis=1)

    x_test = np.expand_dims(self.x_test, axis=2)
    y_test = np.expand_dims(self.y_test, axis=1)
    
    x_val = np.expand_dims(self.x_val, axis=2)
    y_val = np.expand_dims(self.y_val, axis=1)

    return x_train, y_train, x_test, y_test, x_val, y_val

  def processData(self, df_feature, feature_method, normalization = False, standardization = False):
    """

    feature_method is a tuple like:
    (method_name, windows_length)

    """
    feature_method_name = feature_method[0]
    windows_length = feature_method[1]
    if feature_method_name == "RAW":
      processed_data = df_feature
      df_label = self.df['label']
    elif feature_method_name == "SMA":
      processed_data = self.SMA(df_feature, windows_length)
      df_label = self.df['label'].iloc[windows_length-1:].reset_index(drop=True)
    elif feature_method_name == "SMV":
      processed_data = self.SMV(df_feature, windows_length)
      df_label = self.df['label'].iloc[windows_length-1:].reset_index(drop=True)
    elif feature_method_name == "CMA":
      processed_data = self.CMA(df_feature, windows_length)
      df_label = self.df['label'].iloc[windows_length-1:].reset_index(drop=True)
    elif feature_method_name == "CMV":
      processed_data = self.CMV(df_feature, windows_length)
      df_label = self.df['label'].iloc[windows_length-1:].reset_index(drop=True)
    elif feature_method_name == "EMA":
      processed_data = self.EMA(df_feature, windows_length)
      df_label = self.df['label'].iloc[0:].reset_index(drop=True)
    elif feature_method_name == "EMV":
      processed_data = self.EMV(df_feature, windows_length)
      df_label = self.df['label'].iloc[1:].reset_index(drop=True)
    else:
      print("Feature method error")

    if normalization == True and standardization == True:
      print("Normalization and Standartization can not be true at the same time")
    else:
      if normalization:
        processed_data = self.Normalization(processed_data)
      if standardization:
        processed_data = self.Standardization(processed_data)


    print ("process and label\n ")
    print(processed_data)
    print(df_label)
    return processed_data, df_label

  def SMA(self, df_feature, windows_length):
      
    """
    Simple Moving Average
    """
    df_SMA = pd.DataFrame()
    for col in df_feature.columns:
        df_SMA[str(col)] = df_feature[str(col)].rolling(window=windows_length).mean() 
    
    df_SMA.dropna(inplace=True)
    df_SMA = df_SMA.reset_index(drop=True)
    return df_SMA

  def SMV(self, df_feature, windows_length):
      
    """
    Simple Moving Variance
    """
    df_SMV = pd.DataFrame()
    for col in df_feature.columns:
      if col != "label":
        df_SMV[str(col)] = df_feature[str(col)].rolling(window=windows_length).var() 
    
    df_SMV.dropna(inplace=True)
    df_SMV = df_SMV.reset_index(drop=True)

    return df_SMV


  def CMA(self, df_feature, windows_length):
    """
    Cumulative Moving Average
    """
    df_CMA = pd.DataFrame()
    for col in df_feature.columns:
      if col != "label":
        df_CMA[str(col)] = df_feature[str(col)].expanding(min_periods=windows_length).mean() 
    
    df_CMA.dropna(inplace=True)
    df_CMA = df_CMA.reset_index(drop=True)
    return df_CMA
    

      
  def CMV(self, df_feature, windows_length):
    """
    Cumulative Moving Variance
    """
    df_CMV = pd.DataFrame()
    for col in df_feature.columns:
      if col != "label":
        df_CMV[str(col)] = df_feature[str(col)].expanding(min_periods=windows_length).var() 

    df_CMV.dropna(inplace=True)
    df_CMV = df_CMV.reset_index(drop=True)
    return df_CMV


  def EMA(self, df_feature, windows_length):
    """
    Exponential Moving Average
    """
    df_EMA = pd.DataFrame()
    for col in df_feature.columns:
      if col != "label":
        df_EMA[str(col)] = df_feature[str(col)].ewm(span=windows_length, adjust=False).mean() 
    
    df_EMA.dropna(inplace=True)
    df_EMA = df_EMA.reset_index(drop=True)
    return df_EMA

      
  def EMV(self, df_feature, windows_length):
    """
    Exponential Moving Varience
    """
    df_EMV = pd.DataFrame()
    for col in df_feature.columns:
      if col != "label":
        df_EMV[str(col)] = df_feature[str(col)].ewm(span=windows_length, adjust=False).var() 

    df_EMV.dropna(inplace=True)
    df_EMV = df_EMV.reset_index(drop=True)
    return df_EMV

  def Normalization(self, df_feature):
    return pd.DataFrame(MinMaxScaler().fit_transform(df_feature), columns = [x for x in self.headers if x != "label"])

  def Standardization(self, df_feature):
    scaler = StandardScaler()
    scaler.fit(df_feature)
    return pd.DataFrame(scaler.transform(df_feature), columns = [x for x in self.headers if x != "label"])




def main(args):
    test_rate   = args.test_rate  
    validation_rate = args.validation_rate
    headers = args.headers
    label_list = args.label_list
    feature_method = args.feature_method
    normalization = args.normalization
    standardization = args.standardization
    data_file_name = args.data_file_name
    windows_length = args.windows_length
    
    input_data_path = os.path.join("/opt/ml/processing/input", data_file_name)

    data = Data(input_data_path, headers, label_list, (feature_method, windows_length), normalization, standardization, test_rate, validation_rate)

    print (data.processed_data)



if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-rate", type=float, default=0.2)
    parser.add_argument("--validation-rate", type=float, default=0.2)
    parser.add_argument("--headers", type=list, default=[""])
    parser.add_argument("--label-list", type=list, default=[""])
    parser.add_argument("--feature-method", type=str, default=" ")
    parser.add_argument("--windows-length", type=int, default=10)
    parser.add_argument("--normalization", type=bool, default=False)
    parser.add_argument("--standardization", type=bool, default=False)
    parser.add_argument("--data-file-name", type=str, default="input_data")
    
    args, _ = parser.parse_known_args()



    main(args)