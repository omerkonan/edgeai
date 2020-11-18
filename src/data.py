import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
  
  def __init__(self, datapath, headers, seperator, label_list, savepath="./", test_rate=0.2, val_rate=0.2):
    self.datapath = datapath
    self.savepath = savepath
    self.data_type = self.findDataPath()
    self.seperator = seperator
    self.headers = headers
    self.label_list = label_list
    self.df = self.readData()
    self.test_rate = test_rate
    self.val_rate = val_rate/(1-test_rate)
    self.x_train, self.y_train, self.x_test, self.y_test,\
        self.x_val, self.y_val = self.splitData()   
  
  def findDataPath(self):
    filename = self.datapath.split("/")[-1]
    file_extansion = filename.split(".")[-1]
    return file_extansion
  def readData(self):
    '''read data from txt or csv file as a dataframe'''
    if (self.data_type == 'txt'):
      df = pd.read_csv(self.datapath, sep=self.seperator, header= None, engine='c', error_bad_lines=False, warn_bad_lines=False)
      df.columns = self.headers
    elif (self.data_type == 'csv'):
      df = pd.read_csv(self.datapath,engine='c',error_bad_lines=False, warn_bad_lines=False)
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
    x, x_test, y, y_test = train_test_split(self.df[self.df.columns[-3:]], self.df.iloc[:,0],\
                                            test_size=self.test_rate, train_size=(1-self.test_rate))
    x_train, x_val, y_train, y_val = train_test_split(x, y, \
                                            test_size = self.val_rate, train_size =(1-self.val_rate))
    
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