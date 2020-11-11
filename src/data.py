import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

class Data:
  def __init__(self, datapath, headers, seperator, test_rate=0.2, val_rate=0.2):
    self.datapath = datapath
    self.savepath = "./"
    self.data_type = self.findDataPath()
    self.seperator = seperator
    self.headers = headers
    self.df = self.readData()
    self.labels = self.df.iloc[:,:1] # set label column. now it is first column. to set as last column: [:,-1:]  
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
      df = pd.read_csv(self.datapath)
    else: 
      return(print('Error: Datatype error - Wrong DataType input'))
    return df

  def getDataInfo(self):
    return self.df.info()

  def cleanData(self):
    return self.df.dropna(inplace=False)

  def addLabel(self, label):
    labels = self.df['label'] = label
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
  
  def save_df(self, df, fixed_df_path, sep = ','):
    if os.path.exists(fixed_df_path):
    # os.remove(fixed_df_path)
    # Normally this command check file is exist. If it exists, df.to_csv() working as append command.
    # But it occurs premission denied in another drive.
      print("Please delete the fixed dataset and create again")
    else: 
      df.to_csv(fixed_df_path, header=None, index=None, sep=sep)

  def splitData(self):
    x, x_test, y, y_test = train_test_split(self.df, self.labels,\
                                            test_size=self.test_rate, train_size=(1-self.test_rate))
    x_train, x_val, y_train, y_val = train_test_split(x, y, \
                                            test_size = self.val_rate, train_size =(1-self.val_rate))
    return x_train, y_train, x_test, y_test, x_val, y_val

